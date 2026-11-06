"""Read completed development predictions for follow-up planning, never training.

Snapshots and diagnostics live outside the admitted experiment's output tree.
No confirmation/final predictions or model weights are downloaded by this entry.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

import numpy as np

from fusion.next.common import ROOT, OUT, PLAN, digest, identity, read_json, source_identity, write_json
from fusion.next.evaluation import point_scores
from scripts.finish_fusion_100h import SSH_OPTIONS, HOST, check_binding


FOLLOW = ROOT / 'outputs/fusion_followup'
QUARTERS = ('2025Q1', '2025Q3', '2026Q1')
NAMESPACES = ('single_ridge', 'single_lightgbm', 'single_xpatch', 'single_patchtst',
              'simple', 'fusion', 'lora', 'mechanism_plain_mlp')
METADATA = {'cloud/bounded_run_dispatch.json', 'cloud/bounded_run_state.json',
            'cloud/hard_time_limit.json', 'cloud/hard_time_limit_state.json',
            'performance/scope_100h.json', 'performance/portable_inference_policy.json',
            'performance/future_speed_policy.json', 'performance/speed_handoff_result.json'}
MAX_BYTES = 128 * 2**20


def allowed(name):
    parts = PurePosixPath(name).parts
    if name != PurePosixPath(name).as_posix() or '..' in parts or name.startswith('/'):
        return False
    if name in METADATA:
        return True
    if len(parts) == 2 and parts[0] == 'selections':
        return parts[1] in {ns + '.json' for ns in NAMESPACES}
    if len(parts) not in (5, 7) or parts[1] not in NAMESPACES or parts[3] not in QUARTERS:
        return False
    if parts[0] == 'trials':
        if parts[-1] not in ('result.json', 'prediction.npz'):
            return False
        return len(parts) == 5 or (parts[4] in ('seed_2026', 'seed_2027', 'seed_2028')
                                   and parts[5].startswith('updates_')
                                   and parts[5][8:].isdigit())
    return (parts[0] == 'training' and len(parts) == 5
            and parts[-1] in ('seed_2026.progress.json', 'seed_2027.progress.json',
                              'seed_2028.progress.json'))


# Passed on stdin to a read-only remote interpreter; it creates no remote files.
CAPTURE = r'''
import hashlib,io,json,sys,tarfile
from datetime import datetime,timezone
import psutil
from fusion.next.common import OUT,ROOT,PLAN,digest,identity,read_json,source_identity
namespaces=__NAMESPACES__
quarters=__QUARTERS__
metadata=__METADATA__
files={};total=0
def add(name,data):
 global total
 total+=len(data)
 if total>128*2**20:raise ValueError('Development snapshot exceeds its bounded size')
 files[name]=data
for name in metadata:
 p=OUT/name
 if p.exists():add(name,p.read_bytes())
for ns in namespaces:
 p=OUT/'selections'/(ns+'.json')
 if p.exists():add(str(p.relative_to(OUT)),p.read_bytes())
 for p in sorted((OUT/'trials'/ns).rglob('result.json')):
  parts=p.relative_to(OUT).parts
  if len(parts) not in (5,7) or parts[3] not in quarters:continue
  if len(parts)==7 and parts[4] not in ('seed_2026','seed_2027','seed_2028'):continue
  data=p.read_bytes();row=json.loads(data)
  if row['details']['fold']['quarter']!=parts[3]:raise ValueError('Quarter path differs')
  prediction=p.with_name('prediction.npz');values=prediction.read_bytes()
  if hashlib.sha256(values).hexdigest()!=row['prediction_sha256']:raise ValueError('Prediction changed during capture')
  add(str(p.relative_to(OUT)),data);add(str(prediction.relative_to(OUT)),values)
 for p in sorted((OUT/'training'/ns).rglob('*.progress.json')):
  parts=p.relative_to(OUT).parts
  if len(parts)==5 and parts[3] in quarters and parts[-1] in ('seed_2026.progress.json','seed_2027.progress.json','seed_2028.progress.json'):
   add(str(p.relative_to(OUT)),p.read_bytes())
def read(name):
 p=OUT/name
 return read_json(p) if p.exists() else None
def alive(row):
 try:
  p=psutil.Process(row['pid'])
  return abs(p.create_time()-row['created_unix'])<.01 and p.is_running() and p.status()!='zombie'
 except psutil.NoSuchProcess:return False
dispatch=read('cloud/bounded_run_dispatch.json');limiter=read('cloud/hard_time_limit_state.json')
snapshot={'dispatch':dispatch,'state':read('cloud/bounded_run_state.json'),
 'limit':read('cloud/hard_time_limit.json'),'limiter':limiter,
 'numerical_policy':read('performance/portable_inference_policy.json'),
 'receipt':read('cloud/return_receipt.json'),'runner_alive':alive(dispatch),'limiter_alive':alive(limiter)}
manifest={'captured_at_utc':datetime.now(timezone.utc).isoformat(),
 'source_identity':source_identity(),'plan_sha256':digest(PLAN),'snapshot':snapshot,
 'scope_identity':read('performance/scope_100h.json')['identity'],
 'purpose':'post-selection development diagnosis only; not an independent evaluation',
 'Q2_Q3_predictions_included':False,'training_launched':False,
 'files':{name:{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)} for name,data in files.items()}}
manifest['identity']=identity(manifest)
files['manifest.json']=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for name,data in files.items():
  entry=tarfile.TarInfo(name);entry.size=len(data)
  archive.addfile(entry,io.BytesIO(data))
'''


def unpack(archive, destination, binding):
    """Validate the complete allowlist and identities before writing any input."""
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        names = [m.name for m in members]
        if (len(names) != len(set(names)) or 'manifest.json' not in names
                or any(not m.isfile() or (m.name != 'manifest.json' and not allowed(m.name)) for m in members)
                or sum(m.size for m in members) > MAX_BYTES + 2**20):
            raise ValueError('Unexpected, duplicate or oversized development snapshot member')
        manifest = json.load(stream.extractfile('manifest.json'))
        if manifest['identity'] != identity({k: v for k, v in manifest.items() if k != 'identity'}):
            raise ValueError('Snapshot manifest identity changed')
        if set(names) != {'manifest.json', *manifest['files']}:
            raise ValueError('Snapshot manifest member set differs')
        if (manifest['source_identity'] != binding['source_identity']
                or manifest['plan_sha256'] != binding['plan_sha256']
                or manifest['scope_identity'] != binding['scope_identity']
                or manifest['Q2_Q3_predictions_included'] or manifest['training_launched']):
            raise ValueError('Snapshot experiment or information scope differs')
        check_binding(manifest['snapshot'], binding)
        values = {}
        for name, row in manifest['files'].items():
            data = stream.extractfile(name).read()
            if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                raise ValueError('Snapshot file hash differs: ' + name)
            values[name] = data
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in values.items():
        p = destination / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    write_json(destination / 'manifest.json', manifest)
    return destination


def capture():
    FOLLOW.mkdir(parents=True, exist_ok=True)
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = FOLLOW / (tag + '.tar.partial')
    remote = (CAPTURE.replace('__NAMESPACES__', repr(NAMESPACES))
              .replace('__QUARTERS__', repr(QUARTERS)).replace('__METADATA__', repr(sorted(METADATA))))
    command = ('cd /root/autodl-tmp/aapl-fusion\n'
               'export PATH="/root/miniconda3/bin:$PATH"\n'
               ".venv/bin/python - <<'PY_REMOTE_DEVELOPMENT'\n" + remote + '\nPY_REMOTE_DEVELOPMENT\n')
    with archive.open('xb') as output:
        subprocess.run(['ssh', '-p', '19046', *SSH_OPTIONS, HOST, 'bash -s'], input=command.encode(),
                       stdout=output, stderr=subprocess.PIPE, check=True, timeout=60)
    binding = read_json(OUT / 'cloud/local_completion/binding.json')
    destination = unpack(archive, FOLLOW / 'snapshots' / tag, binding)
    archive.unlink()
    return destination


def metrics(target, prediction):
    score = point_scores(target, prediction)
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    zero = score['zero_mse']
    score.update(improvement_pct=100 * score['mse_improvement_vs_zero'],
                 prediction_rms_bps=float(np.sqrt(np.mean(p*p)) * 10000),
                 mean_prediction_bps=float(p.mean()*10000),
                 mean_target_bps=float(y.mean()*10000),
                 forecast_energy_pct_zero=float(100*np.mean(p*p)/zero),
                 target_alignment_pct_zero=float(200*np.mean(p*y)/zero))
    return score


def forward_shrink(parts):
    """Exploratory scalar fit uses strictly earlier quarters, never future blocks."""
    previous_y, previous_p, reports = [], [], []
    for quarter in QUARTERS:
        y, p = parts[quarter]
        if previous_y:
            train_y, train_p = np.concatenate(previous_y), np.concatenate(previous_p)
            energy = float(train_p @ train_p)
            alpha = float(np.clip((train_p @ train_y)/energy, 0, 1)) if energy else 0.
            reports.append({'quarter': quarter, 'fit_quarters': list(QUARTERS[:QUARTERS.index(quarter)]),
                            'alpha': alpha, 'raw': metrics(y, p), 'shrunk': metrics(y, alpha*p)})
        previous_y.append(y)
        previous_p.append(p)
    return {'role': 'post-selection forward diagnostic; no model replacement or independent success claim',
            'quarters': reports}


def validate_rows(rows, prediction, dates, quarter):
    if (quarter not in QUARTERS or rows.ndim != 1 or rows.dtype.kind not in 'iu'
            or not len(rows) or prediction.shape != rows.shape or np.any(np.diff(rows) <= 0)
            or rows.min() < 0 or rows.max() >= len(dates) or not np.isfinite(prediction).all()):
        raise ValueError('Invalid or incomplete development predictions')
    row_dates = dates[rows]
    year, index = int(quarter[:4]), int(quarter[-1])
    start = f'{year}-{3*index-2:02d}-01'
    end = f'{year+1}-01-01' if index == 4 else f'{year}-{3*index+1:02d}-01'
    if np.any((row_dates < start) | (row_dates >= end)):
        raise ValueError('Prediction origin leaves the allowed development quarter')


def analyze(snapshot):
    snapshot = snapshot.resolve()
    if not snapshot.is_relative_to(FOLLOW / 'snapshots'):
        raise ValueError('Analyze only isolated follow-up snapshots')
    manifest = read_json(snapshot / 'manifest.json')
    if (manifest['identity'] != identity({k: v for k, v in manifest.items() if k != 'identity'})
            or manifest['source_identity'] != source_identity() or manifest['plan_sha256'] != digest(PLAN)
            or manifest.get('Q2_Q3_predictions_included') is not False
            or manifest.get('training_launched') is not False):
        raise ValueError('Snapshot provenance differs')
    for name, row in manifest['files'].items():
        if not allowed(name) or digest(snapshot/name) != row['sha256'] or (snapshot/name).stat().st_size != row['bytes']:
            raise ValueError('Development snapshot changed: ' + name)
    dates = np.load(OUT/'data/dates.npy', mmap_mode='r')
    targets = np.load(OUT/'data/AAPL/target.npy', mmap_mode='r')
    stamps = np.load(OUT/'data/timestamps_ns.npy', mmap_mode='r')
    policy = read_json(snapshot/'performance/portable_inference_policy.json')['identity']
    groups, common, prefix_scores = defaultdict(dict), {}, defaultdict(dict)
    for name in sorted(manifest['files']):
        if not name.startswith('trials/') or not name.endswith('/result.json'):
            continue
        result = read_json(snapshot/name)
        details = result['details']
        if details.get('source', manifest['source_identity']) != manifest['source_identity']:
            raise ValueError('Completed prediction belongs to a different engine')
        _, ns, candidate, quarter, *rest = PurePosixPath(name).parts
        if details['fold']['quarter'] != quarter:
            raise ValueError('Prediction quarter metadata differs')
        path = (snapshot/name).with_name('prediction.npz')
        if digest(path) != result['prediction_sha256']:
            raise ValueError('Prediction differs from completed receipt')
        with np.load(path, allow_pickle=False) as values:
            rows, prediction = values['rows'].copy(), values['prediction'].copy()
        validate_rows(rows, prediction, dates, quarter)
        if quarter in common and not np.array_equal(common[quarter], rows):
            raise ValueError('Development methods do not use identical origins')
        common[quarter] = rows
        # Values are indexed only after checking the quarter. No Q2/Q3 labels.
        y = np.asarray(targets[rows], dtype=np.float64)
        score = metrics(y, prediction)
        if (not np.isclose(score['mse'], result['scores']['mse'], rtol=1e-12, atol=0)
                or not np.isclose(score['zero_mse'], result['scores']['zero_mse'], rtol=1e-12, atol=0)):
            raise ValueError('Saved scores disagree with local development targets')
        if ns in ('fusion', 'lora', 'mechanism_plain_mlp') and (
                details.get('evaluation_precision') != 'float32'
                or details.get('numerical_policy_identity') != policy):
            raise ValueError('Neural development prediction lacks the fixed FP32 policy')
        seed = int(details.get('seed', 2026))
        config = details['config']
        updates, total = details.get('updates'), config.get('total_updates')
        if updates is not None and updates != total:
            prefix_scores[(ns, candidate, seed)][quarter+'/'+str(updates)] = {
                'updates': updates, 'total_updates': total,
                'precision': details.get('evaluation_precision', 'historical'),
                'improvement_pct': score['improvement_pct']}
            continue
        groups[(ns, candidate, seed)][quarter] = (y, prediction.astype(np.float64))
    systems, incomplete = {}, []
    all_seeds = defaultdict(dict)
    for (ns, candidate, seed), parts in sorted(groups.items()):
        label = f'{ns}/{candidate}/seed_{seed}'
        if set(parts) != set(QUARTERS):
            incomplete.append({'model': label, 'completed_quarters': sorted(parts)})
            continue
        systems[label] = describe(parts)
        all_seeds[(ns, candidate)][seed] = parts
    for (ns, candidate), seeds in sorted(all_seeds.items()):
        if set(seeds) == {2026, 2027, 2028}:
            parts = {q: (seeds[2026][q][0], np.stack([seeds[s][q][1] for s in (2026,2027,2028)]).mean(0)) for q in QUARTERS}
            systems[f'{ns}/{candidate}/three_seed_mean'] = describe(parts)
    # Describe time-of-day after fixed groups; no selecting a profitable subwindow.
    import pandas as pd
    for model, summary in systems.items():
        ns, candidate, seed_label = model.split('/')
        if seed_label == 'three_seed_mean':
            seeds = all_seeds[(ns, candidate)]
            parts = {q: (seeds[2026][q][0], np.stack([seeds[s][q][1] for s in (2026,2027,2028)]).mean(0)) for q in QUARTERS}
        else:
            parts = groups[(ns, candidate, int(seed_label[5:]))]
        joined_rows = np.concatenate([common[q] for q in QUARTERS])
        clock = pd.to_datetime(stamps[joined_rows], utc=True).tz_convert('America/New_York')
        minute = clock.hour.to_numpy()*60 + clock.minute.to_numpy()
        y = np.concatenate([parts[q][0] for q in QUARTERS])
        p = np.concatenate([parts[q][1] for q in QUARTERS])
        masks = {'opening_0931_1029': minute < 630,
                 'midday_1030_1429': (minute >= 630) & (minute < 870),
                 'closing_1430_1548': minute >= 870}
        summary['time_of_day'] = {name: metrics(y[mask], p[mask]) for name,mask in masks.items() if mask.any()}
    progress = []
    for name in sorted(manifest['files']):
        if name.startswith('training/'):
            row = read_json(snapshot/name)
            progress.append({'path': name, 'step': row['step'], 'total_updates': row['total_updates'],
                             'wall_seconds': row['wall_seconds'], 'loss_trace': row.get('loss_trace', []),
                             'updated_at_utc': row.get('updated_at_utc')})
    report = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'snapshot_identity': manifest['identity'], 'snapshot_time_utc': manifest['captured_at_utc'],
              'source_identity': manifest['source_identity'], 'scope_identity': manifest['scope_identity'],
              'purpose': manifest['purpose'], 'official_experiment_changed': False, 'Q2_Q3_values_read': False,
              'quarters': list(QUARTERS), 'common_origins': {q:len(rows) for q,rows in common.items()},
              'systems': systems, 'incomplete_systems_not_scored_as_complete': incomplete,
              'prefix_scores': {'/'.join(map(str,key)):value for key,value in prefix_scores.items()},
              'training_progress': progress,
              'limits': ['All scores are development diagnostics after candidate selection.',
                         'Forward shrink uses only earlier listed quarters but candidate selection already used development scores.',
                         'Partial seed sets are not treated as three-seed ensembles.',
                         'Training loss alone cannot establish underfitting, overfitting or a benefit from longer training.',
                         'No Chronos attribution is available without a matched no-foundation control.']}
    write_json(snapshot/'analysis.json', report)
    lines = ['# 后续实验开发期诊断', '',
             f'快照 UTC：{report["snapshot_time_utc"]}；身份：`{manifest["identity"]}`。', '',
             '仅分析三个已知开发季度。以下不是独立检验，也不更改本轮选择、冻结或部署。', '',
             '| 模型 | 相对零收益 MSE 改善 % | 预测 RMS / bp | 预测均值 / bp | 改善季度数 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for label,row in sorted(systems.items()):
        s=row['pooled']
        lines.append(f'| {label} | {s["improvement_pct"]:+.6f} | {s["prediction_rms_bps"]:.4f} | {s["mean_prediction_bps"]:.4f} | {sum(x["improvement_pct"]>0 for x in row["quarters"].values())}/3 |')
    lines += ['', '所有预测均通过结果收据哈希、共同起点和本地开发标签重算。详细误差分解、前向收缩、分时段和训练轨迹见同目录 analysis.json。',
              '', 'Q2/Q3 分数、行情、模型权重未由本工具获取；下一轮训练尚未启动。']
    (snapshot/'analysis.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(FOLLOW/'latest_review.json', {'snapshot':str(snapshot.relative_to(ROOT)),
               'snapshot_identity':manifest['identity'], 'analysis':str((snapshot/'analysis.json').relative_to(ROOT)),
               'generated_at_utc':report['generated_at_utc']})
    return report


def describe(parts):
    y = np.concatenate([parts[q][0] for q in QUARTERS])
    p = np.concatenate([parts[q][1] for q in QUARTERS])
    return {'pooled': metrics(y,p), 'quarters': {q: metrics(*parts[q]) for q in QUARTERS},
            'forward_shrink_diagnostic': forward_shrink(parts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--refresh', action='store_true', help='Read a bounded completed-development cloud snapshot')
    group.add_argument('--snapshot', type=Path, help='Recompute an existing isolated snapshot')
    args = parser.parse_args()
    path = capture() if args.refresh else args.snapshot
    report = analyze(path)
    print(json.dumps({'snapshot':str(path), 'snapshot_identity':report['snapshot_identity'],
                      'systems':len(report['systems']), 'training_launched':False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
