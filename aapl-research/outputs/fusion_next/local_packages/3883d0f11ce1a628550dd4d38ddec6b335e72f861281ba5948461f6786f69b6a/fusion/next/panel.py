"""Memory-mapped multistock panel with training-only scaling and balanced day sampling."""
from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd
import torch

from .common import OUT,read_json,digest,identity
from .data import CHANNELS


def close_signals(raw):
    """Close-only day-boundary move is separate from the intraday return series."""
    start=raw[...,7]
    return np.stack([raw[...,11]*(1-start),start,raw[...,9],raw[...,11]*start],axis=-1)


@dataclass
class Panel:
    stocks:list
    background:list
    dates:np.ndarray
    stamps:pd.DatetimeIndex
    features:dict
    observed:dict
    origins:dict
    targets:dict

    def __post_init__(self):
        clock=self.stamps.tz_convert('America/New_York')
        self.clock_minute=(clock.hour.to_numpy()*60+clock.minute.to_numpy()-570).astype(np.int16)
        self.session_ids=np.cumsum(np.r_[True,self.dates[1:]!=self.dates[:-1]])-1

    @classmethod
    def load(cls):
        admission=read_json(OUT/'data_admission.json')
        if not admission['passed']:raise ValueError('Data has not passed F1')
        root=OUT/'data'
        stocks=admission['admitted_training_stocks']
        background=admission['background']+admission['market_peers']
        names=list(dict.fromkeys(stocks+background))
        dates=np.load(root/'dates.npy',mmap_mode='r');stamps=pd.to_datetime(np.load(root/'timestamps_ns.npy',mmap_mode='r'),utc=True)
        fields={field:{s:np.load(root/s/f'{field}.npy',mmap_mode='r') for s in names}
                for field in ('features','observed','origins','target')}
        # Require contemporaneous background availability at every scored origin.
        for s in stocks:
            candidates=np.asarray(fields['origins'][s])
            for bg in background:
                obs=fields['observed'][bg];bad=np.r_[0,np.cumsum(~obs)]
                valid=obs[candidates] & (bad[candidates+1]-bad[candidates-239]<=2)
                candidates=candidates[valid]
            fields['origins'][s]=candidates
        return cls(stocks,background,dates,stamps,fields['features'],fields['observed'],fields['origins'],fields['target'])

    def scaler(self,fit_end,only_aapl=False):
        # A separate seasonal volume baseline for each stock and clock minute.
        mask=self.dates<=fit_end
        minute=self.clock_minute
        result={}
        for name in self.features:
            select=mask & self.observed[name]
            values=np.asarray(self.features[name][select],dtype=np.float64)
            seasonal=np.bincount(minute[select],weights=values[:,4],minlength=390)
            counts=np.bincount(minute[select],minlength=390)
            seasonal=np.divide(seasonal,counts,out=np.full(390,values[:,4].mean()),where=counts>0)
            values[:,4]-=seasonal[minute[select]]
            center=values.mean(0);std=values.std(0);std[std<1e-7]=1.
            close_values=close_signals(values);close_center=close_values.mean(0);close_std=close_values.std(0)
            close_std[close_std<1e-7]=1.
            target_ids=self.origins[name][self.dates[self.origins[name]]<=fit_end]
            target=np.asarray(self.targets[name][target_ids]);target=target[np.isfinite(target)]
            if not len(target):raise ValueError('Scaler fit has no eligible training targets')
            target_std=max(float(np.sqrt(np.mean(target**2))),1e-5)
            result[name]={'center':center.astype(np.float32),'std':std.astype(np.float32),
                          'close_center':close_center.astype(np.float32),'close_std':close_std.astype(np.float32),
                          'volume_seasonal':seasonal.astype(np.float32),'target_std':target_std,'fit_end':fit_end}
        return result

    def windows(self,stock_ids,rows,config,scalers,experts=None,device='cpu',cached=None,include_labels=True):
        rows=np.asarray(rows,dtype=np.int64);stock_ids=np.asarray(stock_ids,dtype=np.int64)
        length=config['lookback']
        if length not in (48,96,240):raise ValueError('Lookback is outside the fixed candidate domain')
        if rows.shape!=stock_ids.shape or rows.ndim!=1 or not len(rows):raise ValueError('Rows and stock IDs must be equal nonempty vectors')
        if rows.min()<240 or rows.max()>=len(self.dates):raise ValueError('Origin lacks the common causal history')
        if stock_ids.min()<0 or stock_ids.max()>=len(self.stocks):raise ValueError('Stock ID outside admitted panel')
        if include_labels and rows.max()+12>=len(self.dates):raise ValueError('Training label is unavailable')
        idx=rows[:,None]+np.arange(1-length,1)
        close_only=config.get('information','ohlcv')=='close'
        local=np.empty((len(rows),length,4 if close_only else len(CHANNELS)),dtype=np.float32)
        foundation=np.empty((len(rows),length,2 if close_only else 3),dtype=np.float32)
        sigma=np.empty(len(rows),dtype=np.float32);target_scale=np.empty(len(rows),dtype=np.float32)
        labels=np.empty(len(rows),dtype=np.float32)
        steps=np.empty((len(rows),12),dtype=np.float32) if include_labels else None
        phase=self.clock_minute[rows]/390
        state=np.empty((len(rows),8),dtype=np.float32)
        for sid in np.unique(stock_ids):
            select=np.flatnonzero(stock_ids==sid);name=self.stocks[int(sid)];s=scalers[name]
            raw=np.asarray(self.features[name][idx[select]],dtype=np.float32)
            close=close_signals(raw) if close_only else None
            foundation[select]=close[:,:,[0,3]] if close_only else raw[:,:,[3,5,4]]
            minute=self.clock_minute[idx[select]]
            x=raw.copy();x[:,:,4]-=s['volume_seasonal'][minute]
            local[select]=np.clip((close-s['close_center'])/s['close_std'] if close_only else (x-s['center'])/s['std'],-10,10)
            return_channel=11 if close_only else 3
            past=raw[:,-min(60,length):,return_channel]
            if close_only:past=past*(1-raw[:,-min(60,length):,7])
            scale=np.sqrt(np.mean(past**2,axis=1)*12+1e-12)
            sigma[select]=scale;target_scale[select]=s['target_std']
            if include_labels:
                labels[select]=self.targets[name][rows[select]]
                steps[select]=self.features[name][rows[select,None]+np.arange(1,13),3]
            state[select]=np.column_stack([np.log(scale/s['target_std']).clip(-6,6),
                (close[:,-12:,0] if close_only else raw[:,-12:,3]).sum(1)/scale,past.sum(1)/(scale*np.sqrt(past.shape[1]/12)),
                np.sin(2*np.pi*phase[select]),np.cos(2*np.pi*phase[select]),phase[select],
                np.where(raw[:,-1,7]>0,raw[:,-1,11],0.) if close_only else raw[:,-1,8],raw[:,:,9].mean(1)])
        background=[]
        for name in (self.background if config.get('market_information',True) else []):
            s=scalers[name];raw=np.asarray(self.features[name][idx],dtype=np.float32)
            minute=self.clock_minute[idx]
            raw[:,:,4]-=s['volume_seasonal'][minute]
            background.append(np.clip((close_signals(raw)-s['close_center'])/s['close_std'] if close_only else (raw-s['center'])/s['std'],-10,10))
        market=np.concatenate(background,axis=2) if background else np.empty((len(rows),length,0),dtype=np.float32)
        if experts is None:experts=np.zeros((len(rows),4),dtype=np.float32)
        batch={'local':local,'market':market,'foundation_context':foundation,'sigma12':sigma,
               'state':state,'stock':stock_ids,'experts':np.asarray(experts,dtype=np.float32),
               'target_scale':target_scale,'row':rows,'session_id':self.session_ids[rows]}
        if include_labels:batch.update(target=labels,step_targets=steps)
        if cached is not None:batch['foundation_cached']=np.asarray(cached,dtype=np.float32)
        if not all(np.isfinite(v).all() for k,v in batch.items() if k!='target'):
            raise ValueError('Nonfinite model inputs')
        return {k:torch.as_tensor(v,device=device) for k,v in batch.items()}


class BalancedSampler:
    def __init__(self,panel,fit_end,seed,only_aapl=False,fit_start='2021-01-04',available=None):
        self.rng=np.random.default_rng(seed);self.groups={};self.stocks=[]
        for sid,name in enumerate(panel.stocks):
            if only_aapl and name!='AAPL':continue
            rows=panel.origins[name]
            rows=rows[(panel.dates[rows]<=fit_end)&(panel.dates[rows]>=fit_start)]
            if available is not None:rows=rows[available[sid][rows]]
            dates=panel.dates[rows]
            groups=np.split(rows,np.flatnonzero(dates[1:]!=dates[:-1])+1) if len(rows) else []
            if groups:self.groups[sid]=groups;self.stocks.append(sid)
        if not self.stocks:raise ValueError('No eligible stock/day training groups')

    def sample(self,batch_size,block=4):
        stock=[];rows=[]
        for _ in range(int(np.ceil(batch_size/block))):
            sid=int(self.rng.choice(self.stocks));days=self.groups[sid]
            day=days[int(self.rng.integers(len(days)))];start=int(self.rng.integers(max(1,len(day)-block+1)))
            selected=day[start:start+block]
            if len(selected)<block:selected=np.resize(selected,block)
            stock.extend([sid]*block);rows.extend(selected.tolist())
        return np.asarray(stock[:batch_size]),np.asarray(rows[:batch_size])

    def state_dict(self):return self.rng.bit_generator.state

    def load_state_dict(self,state):self.rng.bit_generator.state=state


class CoverageSampler(BalancedSampler):
    """Shuffle complete stock/day blocks; balance short days by explicit repeats.

    Every eligible stock/row appears in each coverage epoch. Each stock has the
    same number of day slots, and each day slot the same number of examples.
    Extra day slots rotate randomly across epochs when stock day counts differ.
    """
    policy='balanced_complete_coverage_v1'

    def __init__(self,panel,fit_end,seed,only_aapl=False,fit_start='2021-01-04',available=None):
        super().__init__(panel,fit_end,seed,only_aapl,fit_start,available)
        self.seed=int(seed);self.epoch=0;self.cursor=0;self.emitted=0;self.block=4
        self.panel_rows=len(panel.dates);self.max_days=max(map(len,self.groups.values()))
        self.rows_per_day=max(len(day) for days in self.groups.values() for day in days)
        self.rows_per_day=int(np.ceil(self.rows_per_day/self.block))*self.block
        self.epoch_size=len(self.stocks)*self.max_days*self.rows_per_day
        self.unique_total=sum(len(day) for days in self.groups.values() for day in days)
        self.seen=np.zeros(len(panel.stocks)*self.panel_rows,dtype=bool);self.unique_seen=0
        stamp=hashlib.sha256()
        for sid in self.stocks:
            for day in self.groups[sid]:
                stamp.update(np.asarray([sid,len(day)],dtype='<i8').tobytes())
                stamp.update(np.asarray(day,dtype='<i8').tobytes())
        self.universe_identity=stamp.hexdigest();self._build_epoch()

    def _build_epoch(self):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch]))
        rows=[];stocks=[]
        for sid in self.stocks:
            days=self.groups[sid];slots=np.arange(len(days))
            if len(slots)<self.max_days:
                slots=np.r_[slots,rng.choice(len(days),self.max_days-len(days),replace=True)]
            for index in slots:
                day=days[int(index)];offset=int(rng.integers(len(day)))
                # Rotation makes the repeated remainder fair across epochs, while
                # retaining consecutive origins inside most four-row blocks.
                values=np.resize(np.roll(day,offset),self.rows_per_day).reshape(-1,self.block)
                rows.append(values);stocks.extend([sid]*len(values))
        blocks=np.concatenate(rows);permutation=rng.permutation(len(blocks))
        self.epoch_rows=blocks[permutation].reshape(-1)
        self.epoch_stocks=np.repeat(np.asarray(stocks,dtype=np.int16)[permutation],self.block)

    def sample(self,batch_size,block=4):
        if batch_size<1 or block!=self.block:raise ValueError('Coverage batches require a positive size and four-row blocks')
        stocks=[];rows=[];remaining=batch_size
        while remaining:
            if self.cursor==self.epoch_size:
                self.epoch+=1;self.cursor=0;self._build_epoch()
            count=min(remaining,self.epoch_size-self.cursor);stop=self.cursor+count
            stocks.append(self.epoch_stocks[self.cursor:stop]);rows.append(self.epoch_rows[self.cursor:stop])
            self.cursor=stop;remaining-=count
        stocks=np.concatenate(stocks).astype(np.int64);rows=np.concatenate(rows)
        if self.unique_seen<self.unique_total:
            keys=np.unique(stocks*self.panel_rows+rows)
            self.unique_seen+=int(np.count_nonzero(~self.seen[keys]));self.seen[keys]=True
        self.emitted+=batch_size
        return stocks,rows

    def summary(self):
        return {'policy':self.policy,'eligible_unique_windows':self.unique_total,
                'balanced_epoch_examples':self.epoch_size,'emitted_examples':self.emitted,
                'unique_windows_seen':self.unique_seen,'unique_coverage_fraction':self.unique_seen/self.unique_total,
                'completed_coverage_epochs':self.emitted//self.epoch_size,
                'equivalent_exposure_epochs':self.emitted/self.unique_total,
                'repeat_policy':'equal day-slot length and stock day-slot count; extra slots rotate per epoch'}

    def state_dict(self):
        return {'policy':self.policy,'seed':self.seed,'epoch':self.epoch,'cursor':self.cursor,
                'emitted':self.emitted,'unique_seen':self.unique_seen,'seen_bits':np.packbits(self.seen).tobytes(),
                'universe_identity':self.universe_identity}

    def load_state_dict(self,state):
        if state['policy']!=self.policy or state['seed']!=self.seed or state['universe_identity']!=self.universe_identity:
            raise ValueError('Coverage sampler identity differs from saved training data or recipe')
        self.epoch=state['epoch'];self.cursor=state['cursor'];self.emitted=state['emitted']
        self.unique_seen=state['unique_seen'];self.seen=np.unpackbits(np.frombuffer(state['seen_bits'],dtype=np.uint8),count=len(self.seen)).astype(bool)
        if self.unique_seen!=int(np.count_nonzero(self.seen)) or not 0<=self.cursor<=self.epoch_size:
            raise ValueError('Invalid saved coverage state')
        self._build_epoch()
