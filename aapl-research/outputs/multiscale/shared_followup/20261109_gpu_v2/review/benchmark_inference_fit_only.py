"""Time the real eager inference paths using FIT rows only, without scoring.

No model or output is retained. Metric arithmetic uses fit labels purely to
exercise the same synchronized operations as inner-stop prediction.
"""
from pathlib import Path
import json
import argparse
import sys
import time
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE/'code'))
import experiment as exp

parser=argparse.ArgumentParser()
parser.add_argument('--arm',choices=['tied','untied'])
parser.add_argument('--index',type=int)
cli=parser.parse_args()
exp.setup()
data = exp.data_for(5)
permutation = np.random.default_rng(2026).permutation(len(data.splits['fit']))
rows = []
for arm in ((cli.arm,) if cli.arm else ('tied','untied')):
    net = exp.Bundle(480, data.scaler, 2026, arm == 'tied').to(exp.DEVICE)
    net.eval()
    times = {}
    with torch.inference_mode():
        for mode in ('metric', 'collect'):
            durations = []
            for step in range(64):
                if time.time() >= exp.DEADLINE:
                    raise InterruptedError('Authorized deadline')
                positions = permutation[step*128:(step+1)*128]
                torch.cuda.synchronize()
                before = time.perf_counter()
                if mode == 'metric':
                    args, truth = exp.device_batch(data, 'fit', positions)
                else:
                    args = tuple(value.to(exp.DEVICE) for value in exp.input_batch(data, 'fit', positions))
                predictions = net(*args)
                if mode == 'metric':
                    discarded = sum(float((predictions[h][:,-1]-truth[:,h-1]).double().square().sum())
                                    for h in exp.HORIZONS)
                else:
                    discarded = {h: predictions[h].cpu().numpy()*data.ys[:h]+data.ym[:h]
                                 for h in exp.HORIZONS}
                torch.cuda.synchronize()
                if step >= 8:
                    durations.append(time.perf_counter()-before)
            times[mode] = {'measured_steps_after_warmup':len(durations),
                'batch_size':128,'mean_seconds':float(np.mean(durations)),
                'maximum_seconds':max(durations),'samples_seconds':durations}
    rows.append({'arm':arm,'paths':times})
    del net, args, predictions, discarded
    torch.cuda.empty_cache()
value = {'scope':'fit-only timing of eager metric and input-only prediction collection',
    'rows':rows,'random_fit_positions_seed':2026,'market_stop_or_outer_scored':False,
    'Q3_values_read':False,'candidate_weights_retained':False,'GPU':True,
    'device':torch.cuda.get_device_name(),'affinity':sorted(exp.os.sched_getaffinity(0)),
    'experiment_sha256':exp.sha(BASE/'code/experiment.py'),
    'benchmark_source_sha256':exp.sha(Path(__file__)), 'at_utc':exp.now()}
path=BASE/'audit/GPU_INFERENCE_FIT_ONLY_COST.json' if cli.index is None else BASE/'audit/gpu_inference_concurrent_cost'/f'worker_{cli.index}.json'
exp.atomic(path,value)
print(json.dumps(value),flush=True)
