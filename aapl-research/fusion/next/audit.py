"""Recompute F1 claims from prepared arrays without accessing any sealed data."""
import argparse
import numpy as np
import pandas as pd

from .common import OUT, ROOT, PLAN, digest, identity, read_json, write_json, utc_now
from .data import CHANNELS, POLICY, calendar, eligible_origins,pages_frame,causal_transform
from aapl.data import read_source
from pathlib import Path
from .splits import manifest as time_manifest


def audit(require_complete=False):
    prepared=read_json(OUT/'prepared_symbols.json');stamps,dates,start,first,end=calendar()
    root=OUT/'data';checks={};reports={}
    for name,expected in [('timestamps_ns.npy',stamps.as_unit('ns').asi8),('dates.npy',dates),
                          ('session_start.npy',start),('session_end.npy',end)]:
        checks['calendar_'+name]=np.array_equal(np.load(root/name,mmap_mode='r'),expected)
    checks['calendar_strictly_before_Q3']=bool(stamps.max()<pd.Timestamp('2026-07-01T04:00Z'))
    for name in prepared['symbols']:
        directory=root/name;report=read_json(directory/'manifest.json')
        fields={p[:-4]:np.load(directory/p,mmap_mode='r') for p in report['files']}
        if name=='AAPL':
            source_path=Path(report['source']['source_csv_read_only'])
            if digest(source_path)!=report['source']['sha256']:raise ValueError('Original AAPL source changed')
            raw=read_source(source_path)
        else:raw,_=pages_frame(name)
        original_features,original_observed,original_close,_=causal_transform(raw.reindex(stamps),start)
        valid=all(digest(directory/p)==sha for p,sha in report['files'].items())
        origins=eligible_origins(fields['observed'],end)
        labels=np.log(fields['close'][origins+12]/fields['close'][origins]).astype(np.float32)
        coverage=np.mean(fields['observed']);per_quarter=[]
        for q in pd.period_range('2021Q1','2026Q2',freq='Q'):
            mask=(dates>=str(q.start_time.date()))&(dates<=str(q.end_time.date()))
            per_quarter.append(float(fields['observed'][mask].mean()))
        admitted=bool(coverage>=.995 and all(c>=.995 for c in per_quarter))
        stock_checks={'file_hashes':valid,'feature_shape':fields['features'].shape==(len(dates),len(CHANNELS)),
                      'all_features_rebuilt_from_raw':np.array_equal(original_features,fields['features']),
                      'observations_rebuilt_from_raw':np.array_equal(original_observed,fields['observed']),
                      'prices_rebuilt_from_raw':np.array_equal(original_close,fields['close'],equal_nan=True),
                      'features_finite':bool(np.isfinite(fields['features']).all()),
                      'origins_recomputed':np.array_equal(origins,fields['origins']),
                      'raw_log_labels_recomputed':np.array_equal(labels,fields['target'][origins]),
                      'all_labels_same_session':bool(np.all(dates[origins]==dates[origins+12])),
                      'quarter_coverage_policy_recomputed':admitted==report['admitted'],
                      'only_origin_rows_have_labels':int(np.isfinite(fields['target']).sum())==len(origins),
                      'observed_mask_preserved':np.array_equal(fields['features'][:,9],~fields['observed'])}
        reports[name]={'checks':stock_checks,'passed':all(stock_checks.values()),'admitted':admitted,
                       'manifest_sha256':digest(directory/'manifest.json'),
                       'coverage':float(coverage),'origins':len(origins),'minimum_quarter_coverage':min(per_quarter),
                       'possible_actions':report['possible_corporate_action_or_extreme_gap_flags']}
        write_json(OUT/'data_audit_progress.json',{'last_symbol':name,'checked':len(reports),'total_prepared':len(prepared['symbols']),'at_utc':utc_now()})
        del raw,original_features,original_observed,original_close
    complete=not prepared['pending'] and (OUT/'data_admission.json').exists()
    if complete:
        admission=read_json(OUT/'data_admission.json');names=admission['admitted_training_stocks']
        names=list(dict.fromkeys(names+admission['background']+admission['market_peers']))
        checks['admitted_stock_coverage']=all(reports[s]['admitted'] for s in names)
        common=np.load(root/'AAPL/origins.npy');before=len(common)
        for name in admission['background']+admission['market_peers']:
            obs=np.load(root/name/'observed.npy',mmap_mode='r');bad=np.r_[0,np.cumsum(~obs)]
            common=common[obs[common]&(bad[common+1]-bad[common-239]<=2)]
        checks['common_origin_intersection']=np.array_equal(common,np.load(root/'common_aapl_origins.npy'))
        checks['common_origin_manifest_hash']=digest(root/'common_aapl_origins.npy')==admission['origin_file_sha256']
    checks['all_prepared_shards']=all(row['passed'] for row in reports.values())
    value={'audited_at_utc':utc_now(),'complete':complete,'passed':complete and all(checks.values()),
           'partial_checks_passed':all(checks.values()),'checks':checks,'symbols':reports,'pending':prepared['pending'],
           'calendar_sessions':len(np.unique(dates)),'bar_start_range':[stamps.min().isoformat(),stamps.max().isoformat()],
           'time_splits':time_manifest(dates),'plan_sha256':digest(PLAN),
           'preprocessing_source':{name:digest(ROOT/'fusion/next'/name) for name in ('acquire.py','data.py','panel.py','splits.py')},
           'global_files_sha256':{p.name:digest(p) for p in root.glob('*.npy')},
           'Q3_values_read':False}
    write_json(OUT/'data_audit.json',value)
    if not value['partial_checks_passed']:raise ValueError('Prepared-data audit failed; inspect data_audit.json')
    if require_complete and not value['passed']:raise ValueError('F1 remains incomplete; pending symbols listed in data_audit.json')
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    report=audit(args.require_complete)
    print({'complete':report['complete'],'passed':report['passed'],'prepared':len(report['symbols']),'pending':report['pending']})
