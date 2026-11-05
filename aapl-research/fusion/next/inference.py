"""Load frozen neural/CPU experts and the exact three-seed system for local inference."""
from pathlib import Path

import numpy as np
import torch

from .common import OUT,ROOT,digest,read_json,write_json,utc_now,identity
from .checkpoint import changing_state
from .experiment import model_for
from .experts import NAMES,scope_key


def checkpoint_model(panel,path,config,device):
    path=Path(path);receipt=read_json(path.with_suffix(path.suffix+'.json'))
    if digest(path)!=receipt['sha256']:raise ValueError('Frozen checkpoint hash changed')
    state=torch.load(path,map_location='cpu',weights_only=True);recipe=state['recipe']
    if receipt['recipe']!=recipe or recipe['config']!=config:raise ValueError('Frozen model recipe differs')
    if state['step']!=config['total_updates']:raise ValueError('Cannot export an unfinished training prefix')
    model=model_for(panel,config,device).to(device);expected=set(changing_state(model))
    if set(state['model'])!=expected:raise ValueError('Frozen checkpoint is missing changing parameters or buffers')
    incompatible=model.load_state_dict(state['model'],strict=False)
    if incompatible.unexpected_keys:raise ValueError('Frozen checkpoint contains unexpected parameters')
    model.eval()
    extra=state['extra']
    if extra['stocks']!=panel.stocks or extra['background']!=panel.background:raise ValueError('Frozen stock ordering differs')
    scalers={name:{key:np.asarray(value,dtype=np.float32) if isinstance(value,list) else value for key,value in row.items()}
             for name,row in extra['scalers'].items()}
    return model,scalers,recipe


class FrozenSystem:
    def __init__(self,root,device='cpu'):
        self.root=Path(root).resolve();self.manifest=read_json(self.root/'deployment.json');self.device=device
        expected=self.manifest['identity'];body={k:v for k,v in self.manifest.items() if k!='identity'}
        if identity(body)!=expected:raise ValueError('Deployment manifest identity differs')
        for name,row in self.manifest['files'].items():
            path=(self.root/name).resolve()
            if not path.is_relative_to(self.root) or digest(path)!=row['sha256']:raise ValueError('Deployment artifact identity differs: '+name)
        from types import SimpleNamespace
        panel=SimpleNamespace(stocks=self.manifest['stocks'],background=self.manifest['background'])
        self.stocks=panel.stocks;self.background=panel.background
        self.experts={};self.models=[]
        for name,entry in self.manifest['experts'].items():
            cfg=entry['config']
            if name in ('xpatch','patchtst'):
                model,scalers,_=checkpoint_model(panel,self.root/entry['checkpoint'],cfg,device)
                self.experts[name]=(model,scalers,cfg)
            else:
                directory=self.root/entry['directory'];summary=read_json(directory/'summary.json')
                scalers=read_json(directory/'scalers.json')
                scalers={s:{k:np.asarray(v,dtype=np.float32) if isinstance(v,list) else v for k,v in row.items()} for s,row in scalers.items()}
                unit=summary['target_unit']
                if name=='ridge':
                    with np.load(directory/'ridge.npz') as values:coefficient=values['coefficient'].copy()
                    predict=lambda x,c=coefficient,u=unit:x@c*u
                else:
                    import lightgbm as lgb
                    booster=lgb.Booster(model_file=str(directory/'lightgbm.txt'))
                    predict=lambda x,b=booster,u=unit:b.predict(x,num_threads=2)*u
                self.experts[name]=(predict,scalers,cfg)
        if set(self.experts)!=set(NAMES):raise ValueError('Deployment must contain the four fixed experts')
        if len(self.manifest['system_checkpoints'])!=3:raise ValueError('Deploy exactly the scored three-seed mean system')
        for entry in self.manifest['system_checkpoints']:
            model,scalers,recipe=checkpoint_model(panel,self.root/entry['checkpoint'],entry['config'],device)
            self.models.append((model,scalers,entry['config']))
        self.model_identity=self.manifest['identity']

    @torch.inference_mode()
    def predict_origin(self,panel,rows):
        rows=np.asarray(rows,dtype=np.int64);stocks=np.full(len(rows),self.stocks.index('AAPL'),dtype=np.int64)
        experts=np.empty((len(rows),4),dtype=np.float32)
        from .expert_models import tabular
        for column,name in enumerate(NAMES):
            model,scalers,cfg=self.experts[name]
            batch=panel.windows(stocks,rows,cfg,scalers,device=self.device,include_labels=False)
            if name in ('ridge','lightgbm'):values=np.asarray(model(tabular(batch,len(self.stocks))))
            else:
                with torch.autocast(self.device,dtype=torch.bfloat16,enabled=cfg.get('precision')=='bf16'):
                    values=model(batch)['prediction'].float().cpu().numpy()
            experts[:,column]=values
        seeds=[];weights=[]
        for model,scalers,cfg in self.models:
            batch=panel.windows(stocks,rows,cfg,scalers,experts,device=self.device,include_labels=False)
            with torch.autocast(self.device,dtype=torch.bfloat16,enabled=cfg.get('precision')=='bf16'):
                output=model(batch)
            seeds.append(output['prediction'].float().cpu().numpy());weights.append(output['weights'].float().cpu().numpy())
        # Use the same float64 seed mean as development seed_system(), including
        # local deployment and the two fixed confirmation/final evaluations.
        prediction=np.asarray(seeds,dtype=np.float64).mean(axis=0);weight=np.mean(weights,axis=0)
        if not np.isfinite(prediction).all():raise FloatingPointError('Nonfinite frozen ensemble output')
        return {'prediction':prediction,'weights':weight,'seed_predictions':np.stack(seeds),'expert_predictions':experts}


def export(destination,stocks,background,experts,systems,freeze_sha256):
    """Copy exact fitted checkpoints and CPU artifacts with individually checked hashes."""
    import shutil
    destination=Path(destination).resolve()
    if not destination.is_relative_to(OUT/'deployment'):raise ValueError('Deployment must stay in its dedicated output tree')
    if destination.exists():raise ValueError('Frozen deployment directory already exists')
    destination.mkdir(parents=True);manifest={'stocks':stocks,'background':background,'experts':{},'system_checkpoints':[],
         'freeze_sha256':freeze_sha256,'primary_aggregation':'mean predictions of seeds 2026/2027/2028',
         'files':{},'created_at_utc':utc_now()}
    def copy(path,relative):
        path=Path(path);target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        manifest['files'][relative]={'sha256':digest(target),'bytes':target.stat().st_size}
    def neural(path,relative,config):
        # Retain the exact checkpoint here so recipe/hash verification uses the same
        # recovery artifact; optional optimizer stripping requires a separate schema.
        copy(path,relative);copy(Path(path).with_suffix('.pt.json'),relative+'.json')
        return {'checkpoint':relative,'config':config}
    for name,entry in experts.items():
        if name in ('xpatch','patchtst'):
            manifest['experts'][name]=neural(entry['checkpoint'],'experts/'+name+'.pt',entry['config'])
        else:
            directory=Path(entry['directory']);relative='experts/'+name
            for filename in ('summary.json','scalers.json','ridge.npz' if name=='ridge' else 'lightgbm.txt'):
                copy(directory/filename,relative+'/'+filename)
            manifest['experts'][name]={'directory':relative,'config':entry['config']}
    for index,entry in enumerate(systems):
        manifest['system_checkpoints'].append(neural(entry['checkpoint'],f'system/seed_{2026+index}.pt',entry['config']))
    manifest['identity']=identity(manifest);write_json(destination/'deployment.json',manifest);return manifest
