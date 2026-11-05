"""Budgeted deterministic neural training with continuous halving and early-history pretraining."""
import math
from pathlib import Path
import time

import numpy as np
import torch

from . import checkpoint
from .common import OUT, PLAN, digest, identity, source_identity, write_json, utc_now
from .common import read_json
from .hardware import ResourceGuard
from .panel import BalancedSampler
from .cache import RepresentationCache


class ExpertStore:
    def __init__(self,panel,information='ohlcv'):
        self.arrays={};self.available={};self.sources={}
        import pandas as pd
        for quarter in pd.period_range('2023Q1','2026Q1',freq='Q'):
            receipt=read_json(OUT/'oof'/information/(str(quarter)+'.json'))
            f=receipt['fold']
            if not f['fit_end']<f['purged_full_session']<f['score_start']:
                raise ValueError('OOF training/purge separation invalid')
            if set(receipt['stocks'])!=set(panel.stocks):raise ValueError('Completed OOF quarter lacks admitted stocks')
            for name,row in receipt['stocks'].items():
                if digest(OUT/'oof'/information/row['quarter_file'])!=row['sha256']:
                    raise ValueError('Saved OOF quarter prediction identity differs')
        for sid,name in enumerate(panel.stocks):
            path=OUT/'oof'/information/f'{name}.npy'
            if not path.exists():raise ValueError('OOF expert predictions are not ready for '+name)
            array=np.load(path,mmap_mode='r')
            if array.shape!=(len(panel.dates),4):raise ValueError('OOF array shape differs from development calendar')
            self.arrays[sid]=array;self.available[sid]=np.isfinite(array).all(1)
            dates=panel.dates[self.available[sid]]
            if np.any((dates<'2023-01-01')|(dates>'2026-03-31')):
                raise ValueError('OOF array contains early in-sample or confirmation/final-period predictions')
            self.sources[name]=digest(path)

    def predict(self,stock,rows):
        result=np.empty((len(rows),4),dtype=np.float32)
        for sid in np.unique(stock):
            select=stock==sid;result[select]=self.arrays[int(sid)][rows[select]]
        if not np.isfinite(result).all():raise ValueError('Attempt to train on missing/out-of-scope OOF experts')
        return result


def optimizer_for(model,config,device):
    ordinary=[];adapters=[]
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:(adapters if 'lora_' in name else ordinary).append(parameter)
    groups=[{'params':ordinary,'lr':config['lr'],'initial_lr':config['lr']}]
    if adapters:groups.append({'params':adapters,'lr':config.get('backbone_lr',config['lr']/10),
                              'initial_lr':config.get('backbone_lr',config['lr']/10)})
    return torch.optim.AdamW(groups,weight_decay=config.get('weight_decay',.01),
                            fused=config.get('fused_optimizer',False) and device=='cuda')


def loss_for(output,batch,config,pretrain=False):
    prediction=output['auxiliary_prediction'] if pretrain else output['prediction']
    if config.get('loss')=='step_mse':
        if pretrain or 'steps' not in output:raise ValueError('Step loss requires a supervised 12-step expert')
        return ((output['steps']-batch['step_targets'])/config['raw_loss_unit']).square().mean()
    unit=batch['sigma12'].clamp_min(1e-5) if config.get('loss','raw')=='scaled' else batch['target_scale']
    # A constant training-fit RMS is only a numerical multiplier for raw-MSE optimization.
    if config.get('loss','raw')=='raw':unit=unit.new_full(unit.shape,config['raw_loss_unit'])
    loss=((prediction-batch['target'])/unit).square().mean()
    if not pretrain and 'weights' in output:
        rows=batch['row'];same=(batch['stock'][1:]==batch['stock'][:-1])&(rows[1:]==rows[:-1]+1)
        if same.any():loss=loss+config.get('gate_smooth_penalty',1e-4)*(output['weights'][1:][same]-output['weights'][:-1][same]).square().mean()
    return loss


class Trainer:
    def __init__(self,panel,model,config,fit_end,trial_id,seed=2026,experts=None,device='cuda'):
        self.panel=panel;self.model=model.to(device);self.config=dict(config);self.device=device
        self.scalers=panel.scaler(fit_end);self.fit_end=fit_end;self.experts=experts
        self.config['raw_loss_unit']=self.scalers['AAPL']['target_std']
        only=config.get('training_scope','multistock')=='aapl'
        fusion=config.get('family')=='fusion'
        total=int(config['total_updates']);self.pretrain_updates=(total+4)//5 if fusion else 0
        self.samplers={}
        if fusion:
            if experts is None:raise ValueError('Fusion training requires real earlier-quarter OOF experts')
            self.samplers['pretrain']=BalancedSampler(panel,min(fit_end,'2022-12-31'),seed+1,only)
            self.samplers['main']=BalancedSampler(panel,fit_end,seed+2,only,fit_start='2023-01-01',available=experts.available)
            if config.get('adaptation_fraction',0)>0:
                self.samplers['adapt_pretrain']=BalancedSampler(panel,min(fit_end,'2022-12-31'),seed+3,True)
                self.samplers['adapt_main']=BalancedSampler(panel,fit_end,seed+4,True,fit_start='2023-01-01',available=experts.available)
        else:self.samplers['main']=BalancedSampler(panel,fit_end,seed+2,only)
        self.optimizer=optimizer_for(self.model,config,device);self.step=0
        self.cache=None
        if config.get('representation_cache',False):
            if not fusion or not model.use_foundation:raise ValueError('Representation cache requires a foundation branch')
            self.cache=RepresentationCache(panel,model.foundation,config,storage_dtype=config.get('cache_dtype','float32'))
        self.path=OUT/'training'/trial_id/f'seed_{seed}.pt'
        self.recipe={'config':config,'fit_end':fit_end,'seed':seed,'plan_sha256':digest(PLAN),
                     'source_identity':source_identity(),'scalers_identity':identity({name:{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in row.items()} for name,row in self.scalers.items()}),
                     'expert_sources':experts.sources if experts is not None else {},
                     'pretrain_updates':self.pretrain_updates,'total_updates':total}
        if fusion and model.use_foundation:self.recipe['foundation_weights']=model.foundation.receipt
        self.guard=ResourceGuard(device);self.wall_seconds=0.;self.last_loss=None

    def resume(self):
        self.step,extra=checkpoint.load(self.path,self.model,self.optimizer,self.samplers,self.recipe)
        self.wall_seconds=extra['wall_seconds'];self.last_loss=extra.get('last_loss')

    def save(self):
        checkpoint.save(self.path,self.model,self.optimizer,self.samplers,self.step,self.recipe,
                        {'wall_seconds':self.wall_seconds,'last_loss':self.last_loss,
                         'stocks':self.panel.stocks,'background':self.panel.background,
                         'scalers':{name:{key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in row.items()} for name,row in self.scalers.items()}})

    def run_until(self,updates):
        if not self.step<=updates<=self.config['total_updates']:raise ValueError('Halving must advance a continuous update prefix')
        if not self.path.exists():self.save()
        self.model.train();start=time.perf_counter()
        try:
            while self.step<updates:
                pretrain=self.pretrain_updates>0 and self.step%5==0
                adapting=self.pretrain_updates>0 and self.step>=self.config['total_updates']*(1-self.config.get('adaptation_fraction',0))
                sampler=self.samplers[('adapt_' if adapting else '')+('pretrain' if pretrain else 'main')]
                stocks,rows=sampler.sample(self.config['batch_size'])
                predictions=None if pretrain or self.experts is None else self.experts.predict(stocks,rows)
                batch=self.panel.windows(stocks,rows,self.config,self.scalers,predictions,self.device)
                if self.cache is not None:batch['foundation_cached']=self.cache.get(batch)
                fraction=self.step/self.config['total_updates']
                ratio=self.config.get('adapt_lr_ratio',1.) if adapting else 1.
                for group in self.optimizer.param_groups:group['lr']=ratio*group['initial_lr']*(.1+.9*.5*(1+math.cos(math.pi*fraction)))
                self.optimizer.zero_grad(set_to_none=True)
                bf16=self.config.get('precision','fp32')=='bf16'
                with torch.autocast(device_type=self.device,dtype=torch.bfloat16,enabled=bf16):
                    output=self.model(batch);loss=loss_for(output,batch,self.config,pretrain)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],1.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm')
                self.optimizer.step();self.step+=1;self.last_loss=float(loss.detach())
                if self.step%20==0:self.guard.check()
                if self.step%100==0:
                    elapsed=time.perf_counter()-start;self.wall_seconds+=elapsed;start=time.perf_counter();self.save()
        except BaseException as exc:
            # The last committed prefix is valid even if interruption hits mid-forward or mid-step.
            # Never replace it with partially updated BatchNorm/optimizer/RNG state.
            write_json(self.path.with_suffix('.interruption.json'),{
                'at_utc':utc_now(),'observed_completed_step':self.step,'last_checkpoint_preserved':True,
                'unsaved_attempt_seconds':time.perf_counter()-start,'error_type':type(exc).__name__})
            raise
        else:
            if self.device=='cuda':torch.cuda.synchronize()
            self.wall_seconds+=time.perf_counter()-start;self.save()
        result={'step':self.step,'total_updates':self.config['total_updates'],'last_training_loss':self.last_loss,
                'wall_seconds':self.wall_seconds,'resources':self.guard.check(),'recipe':self.recipe,
                'updated_at_utc':utc_now()}
        write_json(self.path.with_suffix('.progress.json'),result)
        return result

    @torch.inference_mode()
    def predict(self,stocks,rows,expert_predictions=None):
        self.model.eval();results=[]
        for start in range(0,len(rows),self.config['batch_size']):
            stop=start+self.config['batch_size']
            batch=self.panel.windows(stocks[start:stop],rows[start:stop],self.config,self.scalers,
                                      None if expert_predictions is None else expert_predictions[start:stop],self.device,include_labels=False)
            if self.cache is not None:batch['foundation_cached']=self.cache.get(batch)
            with torch.autocast(device_type=self.device,dtype=torch.bfloat16,
                                enabled=self.config.get('precision')=='bf16'):
                value=self.model(batch)['prediction'].float()
            results.append(value.cpu().numpy())
        values=np.concatenate(results)
        if not np.isfinite(values).all():raise FloatingPointError('Nonfinite inference prediction')
        return values
