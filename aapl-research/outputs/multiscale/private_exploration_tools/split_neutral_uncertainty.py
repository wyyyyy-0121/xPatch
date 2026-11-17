"""Descriptive moving-block intervals for post-hoc split-neutral development."""
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from multiscale.aggregate_development import session_labels
from multiscale.bars import file_digest
from multiscale.evaluation import paired_block_bootstrap
from multiscale.spec import MODELS,OUT


def main():
    root=OUT/'private_split_neutral_monthly/20261107T145708694676Z'
    state=json.loads((root/'PRIVATE_QUEUE_STATE.json').read_text())
    plan=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    results_path=root/'PRIVATE_RESULTS.json';results=json.loads(results_path.read_text())
    if (state['status']!='completed_posthoc_development_only' or state['completed']!=5 or
            state['results_sha256']!=file_digest(results_path)):
        raise ValueError('Split-neutral development results are incomplete or changed')
    source=OUT/'admission/alpaca_20261107T052228573242Z';ap=source/'admission.json';a=json.loads(ap.read_text())
    if file_digest(ap)!=plan['admission_sha256'] or a.get('Q3_values_read') is not False:
        raise ValueError('Candidate data identity changed')
    labels=session_labels(source,a,'day');groups=[];diffs={name:[] for name in MODELS};last=None
    for number in range(1,6):
        key=f'fold_{number:02d}';folder=root/key;receipt=json.loads((folder/'fold_receipt.json').read_text())
        if (state['jobs'][key]['status']!='completed' or
                state['jobs'][key]['receipt_sha256']!=file_digest(folder/'fold_receipt.json')):
            raise ValueError('A completed fold changed')
        with np.load(folder/'targets.npz',allow_pickle=False) as x:
            truth=np.asarray(x['actual'][:,-1],dtype=np.float64);orig=x['origins'].copy();dec=x['decision_ns'].copy()
        if last is not None and int(dec[0])<=last:raise ValueError('Fold dates are not increasing')
        last=int(dec[-1]);groups.append(labels[orig])
        with np.load(folder/'prediction_last_close.npz',allow_pickle=False) as x:
            baseline=np.asarray(x['prediction'][:,-1],dtype=np.float64)
        base_loss=np.square(baseline-truth)
        for name in MODELS:
            p=folder/f'prediction_{name}.npz'
            if file_digest(p)!=receipt['models'][name]['prediction_sha256']:raise ValueError('Prediction changed')
            with np.load(p,allow_pickle=False) as x:
                if not np.array_equal(x['origins'],orig) or not np.array_equal(x['decision_ns'],dec):
                    raise ValueError('Prediction origins differ')
                prediction=np.asarray(x['prediction'][:,-1],dtype=np.float64)
            diffs[name].append(np.square(prediction-truth)-base_loss)
    groups=np.concatenate(groups);grid=np.unique(labels);grid=grid[(grid>=groups[0])&(grid<=groups[-1])]
    intervals={}
    for name in MODELS:
        difference=np.concatenate(diffs[name])
        expected=results['models'][name]['endpoint_mse']-results['models']['last_close']['endpoint_mse']
        if not np.isclose(float(difference.mean()),expected,rtol=1e-12,atol=1e-18):
            raise ValueError('Paired loss difference does not match pooled MSE')
        intervals[name]=[]
        for length in (21,42):
            item=paired_block_bootstrap(difference,groups,block_length=length,resamples=5000,
                                        seed=2026,confidence=.95,group_order=grid)
            item.pop('centered_one_sided_p_improvement',None)
            item['status']='descriptive_development_only' if item['bootstrap_status']=='ok' else 'degenerate_bootstrap'
            item['interpretation']='Descriptive paired endpoint loss difference; negative favors the model. Not independent confirmation.'
            intervals[name].append(item)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    parent=OUT/'private_split_neutral_uncertainty';parent.mkdir(parents=True,exist_ok=True)
    directory=parent/stamp;directory.mkdir(exist_ok=False)
    output={'status':'private_posthoc_descriptive_intervals_only','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'task':'monthly H21 split-neutral price log return','folds':5,'validation_origins':len(groups),
            'block_lengths_trading_sessions':[21,42],'resamples':5000,'seed':2026,'confidence':0.95,
            'method':'non-circular moving blocks on full original session grid; paired endpoint squared-loss difference',
            'models':intervals,'source_protocol_sha256':file_digest(root/'PRIVATE_PROTOCOL.json'),
            'source_results_sha256':file_digest(results_path),'script_sha256':file_digest(Path(__file__)),
            'p_values_reported':False,'independent_test':False,'Q3_values_read':False,
            'limitations':['Post-hoc development analysis; intervals do not establish independent significance.',
                           'Overlapping windows, regime changes, model selection and multiple comparisons limit interpretation.',
                           'Price return excludes cash dividends; account-specific publication rights remain unconfirmed.']}
    path=directory/'PRIVATE_DESCRIPTIVE_INTERVALS.json'
    with path.open('x',encoding='utf8') as f:json.dump(output,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    exposure=OUT/'POST_CUTOFF_EXPOSURE.md'
    with exposure.open('a',encoding='utf8') as f:
        f.write(f'\n{stamp}：月度拆股中性开发结果另做事后描述性分块区间，回执 {path}；不作为独立显著性检验。\n')
    print(json.dumps({'status':output['status'],'path':str(path),'sha256':file_digest(path),
                      'models':len(intervals),'validation_origins':len(groups)},ensure_ascii=False))


if __name__=='__main__':main()
