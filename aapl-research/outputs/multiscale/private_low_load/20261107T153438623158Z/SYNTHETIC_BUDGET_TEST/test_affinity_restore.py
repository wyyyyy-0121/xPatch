"""Synthetic-only check of restoring a frozen process's recorded affinity."""
import importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path

script=Path('/home/ayaka/wslcode/aapl-research/outputs/multiscale/private_exploration_tools/night_resume_controller.py')
spec=importlib.util.spec_from_file_location('night_resume_controller_test',script)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
try:
    os.sched_setaffinity(child.pid,{24});os.kill(child.pid,signal.SIGSTOP)
    for _ in range(50):
        raw=Path(f'/proc/{child.pid}/stat').read_text();fields=raw[raw.rfind(')')+2:].split()
        if fields[0] in ('T','t'):break
        time.sleep(.02)
    else:raise RuntimeError('Synthetic process did not stop')
    ticks=int(fields[19]);item={'pid':child.pid,'process_start_ticks':ticks}
    fake={'processes':[{'pid':child.pid,'start_ticks':ticks,'threads':[{'tid':child.pid,'affinity_before':[24,26]}]}]}
    module.restore_thread_affinities(item,fake)
    assert os.sched_getaffinity(child.pid)=={24,26}
    print(json.dumps({'status':'synthetic_affinity_restore_passed','original_affinity_restored':[24,26],
                      'market_data_used':False}))
finally:
    os.kill(child.pid,signal.SIGCONT);child.terminate();child.wait(timeout=5)
