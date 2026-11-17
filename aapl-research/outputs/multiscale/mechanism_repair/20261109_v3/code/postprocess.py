"""Sequential evidence-to-manuscript pipeline; abort visibly on any failed stage."""
from study import *
import subprocess, traceback, time, argparse


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--from-stage')
    args = parser.parse_args()
    assert len(os.sched_getaffinity(0)) <= 2 and os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    assert read(BASE/'audit/CANDIDATE_COMPLETE.json')['jobs'] == 130
    assert read(BASE/'audit/CANDIDATE_CALIBRATION_COMPLETE.json')['jobs'] == 130
    stages = ['analyze_candidate.py', 'figures.py', 'write_manuscripts.py', 'build_documents.py',
              'verify_documents.py', 'build_examples.py', 'reproduce_statistics.py', 'report.py']
    completed = []
    if args.from_stage:
        assert args.from_stage in stages
        previous = read(BASE/'audit/POSTPROCESS_STATE.json')
        completed = previous.get('completed', [])
        stages = stages[stages.index(args.from_stage):]
    for script in stages:
        started = time.time()
        atomic(BASE/'audit/POSTPROCESS_STATE.json', {'status': 'running', 'stage': script,
               'completed': completed, 'pid': os.getpid(), 'at_utc': now()})
        runtime = '/home/ayaka/.venvs/doc-tools/bin/python' if script in \
            ('figures.py', 'write_manuscripts.py', 'build_documents.py', 'verify_documents.py') else sys.executable
        run = subprocess.run([runtime, '-u', str(BASE/'code'/script)],
                             cwd=ROOT, text=True, capture_output=True)
        (BASE/'logs'/(script+'.postprocess.log')).write_text(run.stdout+'\n'+run.stderr)
        if run.returncode:
            atomic(BASE/'audit/POSTPROCESS_ERROR.json', {'stage': script, 'returncode': run.returncode,
                   'stdout_tail': run.stdout[-6000:], 'stderr_tail': run.stderr[-6000:], 'at_utc': now()})
            raise RuntimeError(script+' failed: '+run.stderr[-2000:])
        if script == 'reproduce_statistics.py':
            receipt = json.loads(run.stdout.strip().splitlines()[-1]); assert receipt['status'] == 'passed'
            atomic(BASE/'audit/STATISTICS_REPRODUCED.json', receipt)
        completed.append({'stage': script, 'seconds': time.time()-started})
        print(json.dumps(completed[-1]), flush=True)
    atomic(BASE/'audit/POSTPROCESS_COMPLETE.json', {'status': 'completed_requires_visual_and_release',
           'stages': completed, 'at_utc': now(), 'Q3_values_read': False})
    print('Postprocessing complete; visual review, complete predictor replay and detached package acceptance remain.', flush=True)


if __name__ == '__main__':
    main()
