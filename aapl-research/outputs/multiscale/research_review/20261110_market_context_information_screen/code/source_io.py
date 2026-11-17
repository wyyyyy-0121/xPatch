"""Explicit frozen-cache interface. Import performs no data I/O or market operation."""
from __future__ import annotations
import calendar
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import types
import zipfile
import numpy as np

BAR_FIELDS=('start_ns','end_ns','close','observed','session_date')
VOLUME_FIELDS=BAR_FIELDS+('abnormal','valid')
CUTOFF=calendar.timegm((2026,7,1,4,0,0))*1_000_000_000
MINUTE=60_000_000_000
STAMP=re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:00(?:\.0{1,9})?(?:Z|\+00:00)')

class InputError(ValueError): pass

def digest(body): return hashlib.sha256(body).hexdigest()

def file_sha(path,guard=lambda:None):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            guard(); block=stream.read(65536)
            if not block:return h.hexdigest()
            h.update(block)

class FrozenInputs:
    def __init__(self,refs,guard):
        self.table={}; self.guard=guard; self.audit=[]
        for ref in refs:
            key=str(Path(ref['path']).resolve())
            if key in self.table or not re.fullmatch('[0-9a-f]{64}',str(ref['sha256'])):
                raise InputError('duplicate_or_invalid_frozen_path')
            self.table[key]=ref['sha256']
    def expected(self,path):
        key=str(Path(path).resolve())
        if key not in self.table:raise InputError('input_not_frozen')
        return self.table[key]
    def verify(self,path):
        if file_sha(path,self.guard)!=self.expected(path):raise InputError('input_hash_mismatch')
    def body(self,path):
        self.guard(); expected=self.expected(path); body=Path(path).read_bytes()
        if digest(body)!=expected:raise InputError('parsed_bytes_not_frozen')
        return body
    def json(self,path):return json.loads(self.body(path))
    def module(self,path,name):
        body=self.body(path); module=types.ModuleType(name); module.__file__=str(path)
        exec(compile(body,str(path),'exec'),module.__dict__)
        return module
    def npz(self,path,names):
        body=self.body(path); result={}
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            listing=archive.namelist()
            for name in names:
                self.guard(); member=name+'.npy'
                if listing.count(member)!=1:raise InputError('selected_member_identity_invalid')
                with archive.open(member) as stream:result[name]=np.lib.format.read_array(stream,allow_pickle=False)
                self.audit.append({'path':str(Path(path).resolve()),'member':member})
        return result


def mask_for_history(observed,origins,length):
    result=np.zeros(len(origins),dtype=bool); valid=origins>=length
    missing=np.r_[0,np.cumsum(~observed,dtype=np.int64)]; o=origins[valid]
    result[valid]=missing[o]==missing[o-length]
    return result

class ScreenSource:
    def __init__(self,protocol,rules,guard):
        self.guard=guard; self.rules=rules
        body=Path(protocol['data_references_path']).read_bytes()
        if digest(body)!=protocol['data_references_sha256']:raise InputError('data_references_identity_changed')
        self.refs=json.loads(body)
        self.locked=FrozenInputs([{'path':p,'sha256':h} for p,h in self.refs['dependencies'].items()],guard)
        self.phase='no_labels'; self.bars={}; self.spy={}; self.states=[]
        self.cfg={'source_acceptance':self.refs['source_acceptance_path'],
                  'root_source_acceptance':self.refs['source_root_acceptance_path'],
                  'coverage_acceptance':self.refs['coverage_acceptance_path'],
                  'coverage_results':self.refs['coverage_results_path'],
                  'original_data':next(p for p in self.refs['dependencies'] if p.endswith('/protocol/DATA.json')),
                  'volume_receipt':self.refs['volume_receipt_path'],
                  'context_module':self.refs['old_context_path'], 'volume_paths':self.refs['volume'],
                  'source_index':self.refs['spy_index_path']}
        for key in ('source_acceptance','root_source_acceptance','coverage_acceptance'):
            self.locked.json(self.cfg[key])
        cov=self.locked.json(self.cfg['coverage_results'])
        if (cov['status']!='source_clock_coverage_only_complete' or cov['Q3_read'] is not False
                or cov['features_models_scores_built'] is not False):raise InputError('coverage_not_accepted_clock_only')
        data=self.locked.json(self.cfg['original_data'])
        admission=self.locked.json(Path(data['admission'])/'admission.json')
        self.actions=admission['split_actions']
        receipt=self.locked.json(self.cfg['volume_receipt'])
        if receipt.get('status')!='complete' or receipt.get('Q3_values_read') is not False:
            raise InputError('old_volume_cache_not_complete')
        self.context=self.locked.module(self.cfg['context_module'],'locked_spy_screen_context17')
        for frequency in ('minute','hour'):
            path=self.cfg['volume_paths'][frequency]
            if self.locked.expected(path)!=receipt['cache_sha256'][frequency]:raise InputError('volume_cache_receipt_mismatch')
            bars=self.locked.npz(path,VOLUME_FIELDS); n=len(bars['close'])
            original=self.locked.npz(self.refs['bars'][frequency],BAR_FIELDS)
            if any(not np.array_equal(bars[k],original[k]) for k in BAR_FIELDS):
                raise InputError('cached_volume_original_bars_identity_changed')
            del original
            if (any(a.ndim!=1 or len(a)!=n for a in bars.values()) or bars['close'].dtype!=np.float64
                    or not np.isfinite(bars['close']).all() or (bars['close']<=0).any()
                    or bars['start_ns'].dtype!=np.int64 or bars['end_ns'].dtype!=np.int64
                    or bars['observed'].dtype!=np.bool_ or not bars['observed'].all()
                    or bars['session_date'].dtype.kind!='U' or bars['valid'].dtype!=np.bool_
                    or not np.array_equal(bars['valid'],np.isfinite(bars['abnormal']))
                    or (np.diff(bars['start_ns'])<=0).any() or (np.diff(bars['end_ns'])<=0).any()
                    or (bars['end_ns']>CUTOFF).any()):raise InputError('volume_original_clock_or_fields_invalid')
            bars['log_close']=np.log(bars['close']); bars['log_split']=np.zeros(n)
            previous=None
            for action in self.actions:
                when=str(action['date']); ratio=float(action['ratio'])
                if (previous is not None and when<=previous) or not np.isfinite(ratio) or ratio<=0:
                    raise InputError('split_action_invalid')
                bars['log_split'][bars['session_date']>=when]+=np.log(ratio); previous=when
            bars['abnormal_sum']=np.r_[0.,np.cumsum(np.where(bars['valid'],bars['abnormal'],0.))]
            bars['abnormal_bad']=np.r_[0,np.cumsum(~bars['valid'],dtype=np.int64)]
            self.bars[frequency]=bars
        source_index=self.locked.json(self.cfg['source_index'])
        self._spy_on_original_grid(source_index,protocol)
        by_key={(row['task'],row['fold']):row for row in cov['rows']}
        if len(by_key)!=4:raise InputError('coverage_four_states_required')
        for task in rules['tasks']:
            for fold in task['folds']:
                row=by_key[(task['name'],fold)]; path=row['mask_file']
                if self.locked.expected(path)!=row['mask_file_sha256']:raise InputError('coverage_mask_hash_mismatch')
                saved=self.locked.npz(path,('purged_prefix_origins','purged_prefix_complete_history','unchanged_tail_origins','unchanged_tail_complete_history'))
                self.states.append(SourceState(self,task,fold,saved,row))
    def _spy_on_original_grid(self,index,protocol):
        if (index['kind']!='SPY_qualified_canonical_index_v1' or index['all_67_ranges_complete'] is not True
                or index['Q3_allowed'] is not False or index['protocol_sha256']!=protocol['source_protocol_sha256']
                or index['inheritance_manifest_sha256']!=protocol['inheritance_manifest_sha256']):raise InputError('source_index_identity_invalid')
        stamps=[]; close=[]; previous=None
        for page in index['pages']:
            self.guard(); ref=page['canonical_ref']; path=Path(ref['path'])
            if not 0<ref['bytes']<=4*1024*1024:raise InputError('canonical_page_cap_invalid')
            with path.open('rb') as stream:body=stream.read(ref['bytes']+1)
            if len(body)!=ref['bytes'] or digest(body)!=ref['sha256']:raise InputError('canonical_source_bytes_changed')
            payload=json.loads(body)
            if set(payload['bars'])!={'SPY'}:raise InputError('SPY_only')
            for j,bar in enumerate(payload['bars']['SPY']):
                if j%256==0:self.guard()
                if not STAMP.fullmatch(bar['t']):raise InputError('source_exact_UTC_minute_required')
                stamp=calendar.timegm(datetime.fromisoformat(bar['t'].replace('Z','+00:00')).utctimetuple())*1_000_000_000
                value=bar['c']
                if (stamp>=CUTOFF or (previous is not None and stamp<=previous) or isinstance(value,bool)
                        or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0):raise InputError('source_close_or_clock_invalid')
                stamps.append(stamp);close.append(value);previous=stamp
        stamps=np.asarray(stamps,dtype=np.int64); close=np.asarray(close,dtype=np.float64)
        original=self.bars['minute']['start_ns']; positions=np.searchsorted(stamps,original)
        available=positions<len(stamps); candidates=np.flatnonzero(available)
        available[candidates]=stamps[positions[candidates]]==original[candidates]
        values=np.full(len(original),np.nan); values[available]=close[positions[available]]
        self.spy['minute']=values
        minute=self.bars['minute']; hour=self.bars['hour']; first=np.flatnonzero(np.r_[True,minute['session_date'][1:]!=minute['session_date'][:-1]])
        stop=np.r_[first[1:],len(original)]; starts=[];ends=[];dates=[];prices=[]
        for a,b in zip(first,stop):
            if (np.diff(original[a:b])!=MINUTE).any():raise InputError('minute_session_gap')
            for j in range(int(a),int(b)-59,60):
                starts.append(original[j]);ends.append(minute['end_ns'][j+59]);dates.append(minute['session_date'][j])
                prices.append(values[j+59] if np.isfinite(values[j:j+60]).all() else np.nan)
        if (not np.array_equal(hour['start_ns'],starts) or not np.array_equal(hour['end_ns'],ends)
                or not np.array_equal(hour['session_date'],dates)):raise InputError('original_hour_construction_changed')
        self.spy['hour']=np.asarray(prices,dtype=np.float64)
    def allow_prefix_labels(self):self.phase='prefix'
    def allow_tail_labels(self):self.phase='tail'

class SourceState:
    def __init__(self,source,task,fold,saved,row):
        self.source=source; self.task=task['name'];self.frequency=task['frequency'];self.L=task['L'];self.H=task['H'];self.fold=fold
        self.key=self.task+'_fold_'+str(fold); self.bars=source.bars[self.frequency]
        self.P=saved['purged_prefix_origins']; self.T=saved['unchanged_tail_origins']
        self.mSP=saved['purged_prefix_complete_history']; self.mST=saved['unchanged_tail_complete_history']
        for origins,mask,part in ((self.P,self.mSP,'purged_prefix'),(self.T,self.mST,'unchanged_tail')):
            if (origins.dtype!=np.int64 or origins.ndim!=1 or mask.dtype!=np.bool_ or mask.shape!=origins.shape
                    or (np.diff(origins)<=0).any() or (origins<self.L-1).any() or (origins+self.H>=len(self.bars['close'])).any()
                    or digest(origins.astype('<i8').tobytes())!=row['partitions'][part]['origin_array_sha256']
                    or digest(mask.tobytes())!=row['partitions'][part]['mask_sha256']
                    or not np.array_equal(mask,mask_for_history(np.isfinite(source.spy[self.frequency]),origins,self.L))):raise InputError('original_partition_or_source_mask_changed')
        if len(self.T)==0 or len(self.P)==0 or self.bars['end_ns'][self.P[-1]+self.H]>=self.bars['end_ns'][self.T[0]]:
            raise InputError('original_prefix_label_clock_not_purged')
        self.mVP=self.volume_valid(self.P);self.mVT=self.volume_valid(self.T);self.PV=self.P[self.mVP]
        self.complete_PV=self.mSP[self.mVP]
        names=source.context.feature_names(self.H,self.L)
        if names!=source.rules['features']['context17'][self.frequency]:raise InputError('exact_context17_changed')
        self.lag_column=names.index('return_lag_'+str(self.H))
    def volume_valid(self,origins):
        bad=self.bars['abnormal_bad'];return bad[origins+1]==bad[origins-self.H+1]
    def A(self,origins):
        self.source.guard()
        if origins.ndim!=1 or (origins<self.L-1).any() or (origins>=len(self.bars['close'])).any():
            raise InputError('A_origin_outside_history')
        idx=origins[:,None]+np.arange(1-self.L,1)
        if not self.bars['observed'][idx].all():raise InputError('original_history_unobserved')
        history=self.bars['log_close'][idx]+self.bars['log_split'][idx]-self.bars['log_split'][origins,None]-self.bars['log_close'][origins,None]
        anchor=self.bars['close'][origins,None];old=self.source.context.features(history,self.bars['end_ns'][origins],self.H)
        x=np.column_stack((history,anchor,old))
        if x.shape!=(len(origins),self.L+18) or not np.isfinite(x).all():raise InputError('A_dictionary_invalid')
        return x
    def B(self,origins):
        if not self.volume_valid(origins).all():raise InputError('B_expert_missing_volume_input')
        a=self.A(origins); sums=self.bars['abnormal_sum'];u=(sums[origins+1]-sums[origins-self.H+1])/self.H
        return np.column_stack((a,u,a[:,self.L+1+self.lag_column]*u))
    def z(self,origins):
        if (origins<self.L).any() or (origins>len(self.source.spy[self.frequency])).any():
            raise InputError('SPY_complete_delayed_window_required')
        indices=origins[:,None]+np.arange(-self.L,0); values=self.source.spy[self.frequency][indices]
        if not np.isfinite(values).all():raise InputError('SPY_complete_delayed_window_required')
        return np.diff(np.log(values),axis=1)
    def plus(self,origins,complete,mean):
        z=np.broadcast_to(mean,(len(origins),self.L-1)).copy()
        if complete.any():z[complete]=self.z(origins[complete])
        return np.column_stack((self.B(origins),z))
    def labels(self,origins,phase):
        if self.source.phase!=phase or phase not in ('prefix','tail'):raise InputError('labels_before_phase_admission')
        allowed=self.P if phase=='prefix' else self.T; pos=np.searchsorted(allowed,origins)
        if (pos>=len(allowed)).any() or not np.array_equal(allowed[pos],origins):raise InputError('labels_outside_exact_original_partition')
        idx=origins[:,None]+np.arange(1,self.H+1)
        if not self.bars['observed'][idx].all():raise InputError('target_unobserved')
        if self.frequency=='minute' and not (self.bars['session_date'][origins]==self.bars['session_date'][origins+self.H]).all():raise InputError('minute_target_cross_session')
        return np.log(self.bars['close'][idx]/self.bars['close'][origins,None])
