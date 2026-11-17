"""Recompute released mean losses and paired intervals using local package files only."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from common import PROFILES,MECHANISM,key
from analyze import groups_for,bootstrap,factorial_interaction
from multiscale.bars import Bars

def load(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def intervals_equal(actual,expected):
    if actual['status']!=expected['status']:return False
    if actual.get('supports_improvement')!=expected.get('supports_improvement'):return False
    for field in ('relative_improvement_pct','mse_reference_minus_primary','mean_loss_interaction'):
        if field in actual and not np.isclose(actual[field],expected[field],rtol=1e-12,atol=1e-15):return False
    if actual.get('relative_ci95_pct') is not None:
        if not np.allclose(actual['relative_ci95_pct'],expected['relative_ci95_pct'],rtol=1e-12,atol=1e-12):return False
    if actual.get('difference_ci95') is None or expected.get('difference_ci95') is None:
        return actual.get('difference_ci95')==expected.get('difference_ci95')
    return np.allclose(actual['difference_ci95'],expected['difference_ci95'],rtol=1e-12,atol=1e-15)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=ap.parse_args()
    root=args.root.resolve();result=load(root/'analysis/RESULTS.json');admission=load(root/'data/admission.json')
    bars={}
    for frequency in ('month','day','hour','minute'):
        p=root/'data'/f'bars_{frequency}.npz';assert digest(p)==admission['files'][p.name]
        with np.load(p,allow_pickle=False) as a:bars[frequency]=Bars(frequency,**{k:a[k] for k in a.files})
        assert bars[frequency].session_date[-1]<='2026-06-30'
    expected={(r['task'],r['primary'],r['reference'],r['block']):r for r in result['intervals']}
    points=0;interval_count=0;interactions=0
    for profile,(frequency,lookback,hs) in PROFILES.items():
        for h in hs:
            for target in ('future_close','future_log_return'):
                task=key(profile,h,target);groups=[];sums={};N=0
                for fn in range(1,6):
                    with np.load(root/'analysis/paired_losses'/task/f'fold_{fn}.npz') as a:
                        origins=a['origins'];losses={k[5:]:a[k] for k in a.files if k.startswith('loss_')}
                        assert np.array_equal(a['session_date'],bars[frequency].session_date[origins])
                    groups.append(groups_for(bars[frequency],origins,losses));N+=len(origins)
                    for name,loss in losses.items():sums[name]=sums.get(name,0.)+float(loss.sum())
                for row in result['pooled']:
                    if row['task']!=task:continue
                    assert row['origins']==N and np.isclose(sums[row['variant']]/N,row['endpoint_mse'],rtol=1e-12,atol=1e-15)
                    points+=1
                mechanism=(profile,h,target) in MECHANISM
                if mechanism:
                    pairs=[('long_aligned',name) for name in ('zero','original_lstm','ridge','shrink_lstm','short_path','short_aligned','long_path')]
                    pairs += [('long_path','short_path'),('short_aligned','short_path')]
                    blocks=[1,5] if profile=='minute' else [2,5]
                else:
                    pairs=[('long_aligned','zero' if target=='future_log_return' else 'persistence')]
                    blocks={'annual':[12,24],'quarterly':[3,6],'monthly':[21,42],'hourly':[2,5],'minute':[1,5]}[profile]
                    if profile in ('annual','quarterly','monthly'):
                        groups=[{'names':groups[0]['names'],'counts':np.concatenate([g['counts'] for g in groups]),
                            'sums':np.concatenate([g['sums'] for g in groups])}]
                for block in blocks:
                    rows=bootstrap(groups,pairs,block,task)
                    for row in rows:
                        e=expected[(task,row['primary'],row['reference'],block)]
                        assert intervals_equal(row,e),(task,row['primary'],row['reference'],block)
                        interval_count+=1
                    if mechanism:
                        row=factorial_interaction(groups,block,task)
                        e=next(x for x in result['factorial_interactions'] if x['task']==task and x['block']==block)
                        assert intervals_equal(row,e);interactions+=1
    fusion_points=fusion_intervals=0
    for r in result['fusion']:
        task=r['task'];profile=task.split('_h')[0];frequency=PROFILES[profile][0];groups=[];sums={};N=0
        for fn in (3,4,5):
            with np.load(root/'analysis/fusion_paired_losses'/task/f'fold_{fn}.npz') as a:
                origins=a['origins'];losses={k[5:]:a[k] for k in a.files if k.startswith('loss_')}
            groups.append(groups_for(bars[frequency],origins,losses));N+=len(origins)
            for name,loss in losses.items():sums[name]=sums.get(name,0.)+float(loss.sum())
        for row in r['pooled']:
            assert np.isclose(sums[row['method']]/N,row['endpoint_mse'],rtol=1e-12,atol=1e-15);fusion_points+=1
        if 'gate' in sums:
            for block in ([1,5] if profile=='minute' else [2,5]):
                for row in bootstrap(groups,[('gate',n) for n in ('zero','equal','constant','mlp','gru','lstm','xpatch')],block,task):
                    e=next(x for x in r['intervals'] if x['reference']==row['reference'] and x['block']==block)
                    assert intervals_equal(row,e);fusion_intervals+=1
    receipt={'status':'passed','new_pooled_rows':points,'paired_intervals':interval_count,
        'factorial_interactions':interactions,'fusion_pooled_rows':fusion_points,'fusion_intervals':fusion_intervals,
        'original_workspace_or_network_used':False,'all_data_local_pre_cutoff':True}
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
