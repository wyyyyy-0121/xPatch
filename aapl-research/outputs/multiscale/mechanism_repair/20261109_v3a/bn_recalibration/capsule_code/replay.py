"""Replay archived BN adapters from past-only prepared inputs."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from portable_predictor import Predictor

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    cases=json.loads((ROOT/'CASE_MANIFEST.json').read_text())['cases'];rows=[]
    for row in cases:
        checkpoint=ROOT/row['checkpoint'];path=ROOT/row['file']
        assert digest(checkpoint)==row['checkpoint_sha256'] and digest(path)==row['file_sha256']
        model=Predictor(checkpoint)
        with np.load(path,allow_pickle=False) as a:
            p=model.predict_prepared(a['log_relative_history'],a['current_close'])
            shrink=model.predict_prepared(a['log_relative_history'],a['current_close'],shrink_alpha=row['alpha'])
            assert np.allclose(p,a['expected_raw'],rtol=1e-6,atol=1e-8)
            assert np.allclose(shrink,a['expected_shrink'],rtol=1e-6,atol=1e-8)
            rows.append({'job':row['job'],'origins':len(p),'raw_exact':bool(np.array_equal(p,a['expected_raw'])),
                'shrunk_exact':bool(np.array_equal(shrink,a['expected_shrink'])),
                'raw_max_abs_error':float(np.max(abs(p-a['expected_raw']))),
                'shrunk_max_abs_error':float(np.max(abs(shrink-a['expected_shrink'])))})
    print(json.dumps({'status':'passed','case_count':len(rows),'rows':rows,'targets_opened':False}),flush=True)

if __name__=='__main__':main()
