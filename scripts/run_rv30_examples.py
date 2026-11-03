"""Run real CLI examples without changing model or research artifacts."""
from datetime import datetime,timezone
from pathlib import Path
import json
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from finance.io import ROOT,write_json


def main():
    output=ROOT/'artifacts/rv30/setup/prospective_20261104/examples'
    output.mkdir(parents=True,exist_ok=False)
    commands=[
      ('help',['-m','finance','--help'],0,None),
      ('historical',['-m','finance','infer','--as-of','2026-06-30T10:00:00-04:00','--output',str(output/'historical.json')],0,'historical_replay_candidate'),
      ('closed',['-m','finance','infer','--as-of','2026-06-30T17:00:00-04:00'],2,'outside_session_or_insufficient_history'),
      ('before_model',['-m','finance','infer','--as-of','2022-05-02T10:00:00-04:00'],2,'before_frozen_model_availability'),
      ('live',['-m','finance','infer','--mode','live'],2,'live_unavailable'),
      ('early_forward_verify',['-m','finance','prospective','verify','--round','1'],2,'waiting_60_complete_sessions'),
      ('pending_acceptance_refused',['-m','finance','accept','--run-directory','artifacts/rv30/65a4daf1ebacb8a7dedca5e866010bd94cda25acc7124509ec81d1c14dc2fbf8/development'],1,None),
      ('performance_resume_gate',['scripts/start_rv30_checked.py','--run-name','development','--check-only'],0,None),
      ('performance_new_experiment_refused',['scripts/start_rv30_checked.py','--run-name','unregistered_new_experiment','--check-only'],2,None)
    ]
    records=[]
    for name,args,code,status in commands:
        result=subprocess.run([sys.executable,*args],cwd=ROOT,capture_output=True,text=True,timeout=180)
        (output/(name+'.stdout')).write_text(result.stdout);(output/(name+'.stderr')).write_text(result.stderr)
        reported=None
        if status:
            try:reported=json.loads(result.stdout)['status']
            except (ValueError,KeyError):pass
        passed=result.returncode==code and (status is None or status==reported)
        if name=='pending_acceptance_refused':
            check=json.loads(result.stdout);passed=passed and check['passed'] is False and check['goal_can_complete'] is False
        records.append({'name':name,'command':[sys.executable,*args],'exit_code':result.returncode,
            'expected_exit_code':code,'expected_status':status,'actual_status':reported,'passed':passed,
            'stdout':str((output/(name+'.stdout')).relative_to(ROOT)),'stderr':str((output/(name+'.stderr')).relative_to(ROOT))})
        print(name,passed,flush=True)
    write_json(ROOT/'delivery/cli_examples.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'passed':all(r['passed'] for r in records),'commands':records,'scope':'real_pending_candidate_CLI_not_final_acceptance'})
    return 0 if all(r['passed'] for r in records) else 1
if __name__=='__main__':raise SystemExit(main())
