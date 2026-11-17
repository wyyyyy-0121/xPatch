"""Update an English working draft from hash-checked all-task result tables."""
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path('/home/ayaka/wslcode/aapl-research')
OUT = ROOT / 'outputs/multiscale'
PARENT = OUT / 'private_working_manuscripts/20261107T190006607631Z'
PACKAGE = OUT / 'historical_inference_packages/20261108_all20_tasks'
PACKAGE_AUDIT = OUT / 'historical_inference_packages/20261108_all20_tasks_audit/PACKAGE_AUDIT.json'
RESTART = OUT / 'historical_training_recovery/20261108_real_mid_epoch_verified/RESTART_AUDIT.json'
NIGHT = OUT / 'private_low_load/20261107T153438623158Z/ACTUAL_RESUME_VERIFICATION.json'
MODELS = ('last_close','drift','seasonal','ridge','mlp','gru','lstm','xpatch')
NAMES = ('Persistence / zero','Drift','Seasonal','Ridge','MLP','GRU','LSTM','xPatch')
NEURAL = ('mlp','gru','lstm','xpatch')
SCALES = ('annual','quarterly','monthly','hourly','minute')


def load(path): return json.loads(path.read_text(encoding='utf8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def require(ok, message):
    if not ok: raise ValueError(message)
def replace_once(text, before, after):
    require(text.count(before)==1, 'Missing or ambiguous fragment: '+before[:80])
    return text.replace(before, after, 1)
def section(text, start, end, body):
    require(text.count(start)==text.count(end)==1, 'Missing section boundary')
    a,b=text.split(start,1);_,c=b.split(end,1)
    return a+start+'\n\n'+body.strip()+'\n\n'+end+c
def table(header, rows):
    return '\n'.join(['| '+' | '.join(header)+' |','| '+' | '.join(['---']+['---:']*(len(header)-1))+' |']+
                      ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])
def percent(value): return '—' if value is None else f'{100*value:.2f}%'
def number(value): return '—' if value is None else f'{value:.6g}'
def task_name(task):
    return task['profile'].title()+f" H{task['horizon']}"+(' (short history)' if task['profile']=='annual' else '')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True);args=parser.parse_args()
    source=args.results.resolve();result=load(source/'RESULTS.json');protocol=load(source/'COMPILATION_PROTOCOL.json')
    require(result['protocol_sha256']==sha(source/'COMPILATION_PROTOCOL.json'),'Compiler plan changed')
    for name,expected in result['csv_sha256'].items(): require(sha(source/name)==expected,'CSV changed: '+name)
    require(result['complete_tasks']==12 and len(result['tasks'])==20 and result['p_values_reported'] is False and
            result['Q3_values_read'] is False,'This revision expects the twelve-complete-task snapshot')
    coverage=load(Path(protocol['coverage']));require(sha(Path(protocol['coverage']))==protocol['coverage_sha256'],'Coverage changed')
    require(coverage['completed_folds']==result['completed_folds'],'Coverage count differs')
    restart,package_audit,recovery=map(load,(RESTART,PACKAGE_AUDIT,NIGHT))
    require(restart['status']=='bounded_real_training_fresh_process_restart_verified' and
            restart['fresh_processes']==6 and all(x['all_checkpoint_fields_exactly_equal'] for x in restart['checks']),
            'Real checkpoint restart evidence differs')
    require(recovery['original_remaining_budgets_preserved'] is True,'Budget recovery evidence differs')
    tasks=sorted(result['tasks'],key=lambda t:(SCALES.index(t['profile']),t['horizon'],t['target']))
    h60=next(t for t in tasks if (t['profile'],t['horizon'],t['target'])==('minute',60,'future_log_return'))
    n=result['completed_folds'];complete=result['complete_tasks'];minute_n=n-60
    stamp=datetime.fromisoformat(coverage['created_at_utc']).astimezone(ZoneInfo('Asia/Shanghai')).strftime('%H:%M')
    text=(PARENT/'PRIVATE_WORKING_MANUSCRIPT_EN.md').read_text(encoding='utf8')
    original=text
    status=text.splitlines()[2]
    text=replace_once(text,status,
        f'**WORKING MANUSCRIPT: EXPERIMENTS IN PROGRESS.** This revision reports {n}/100 completed historical development folds. '
        'The researcher has confirmed permission for all Alpaca and Kibot data. The minute matrix remains incomplete; '
        'the model package provides an audited interface for all twenty tasks using the available folds.')
    gru=h60['pooled']['models']['gru']['relative_improvement']
    change=f'{abs(100*gru):.2f}% '+('lower' if gru>0 else 'higher')
    abstract=(
        'Forecast accuracy for one stock can vary across horizons and target representations. We compare persistence, '
        'drift, seasonal repetition, ridge regression, an MLP, GRU, LSTM and an adapted xPatch model on Apple Inc. '
        '(AAPL) prices and cumulative log returns across five temporal scales. '
        f'This historical development snapshot contains {n} of 100 planned folds, with {complete} of 20 tasks complete. '
        'Across the six complete nominal-price tasks, none of 24 neural task–model comparisons has lower pooled '
        'endpoint MSE than persistence. '
        f'The minute H60 return GRU has {change} pooled endpoint MSE than zero return across '
        f'{len(h60["completed_folds"])} available folds. '
        'In a post-hoc split-neutral monthly comparison, the GRU’s relative MSE improvements are 3.96%, -24.67% '
        'and 3.99% across three seeds. The annual task has 47 overlapping validation origins. We provide all-task '
        'descriptive uncertainty tables and a verified historical prediction interface, and preserve missing folds '
        'in the reporting grid. Prior exposure, limited annual history and unfinished minute comparisons restrict '
        'the conclusions; the study supplies no independent confirmation or trading evaluation.')
    text=section(text,'## Abstract','## 1. Introduction',abstract)
    text=replace_once(text,
        'The realized update counts for the 12 completed tasks are summarized in Section 5; the remaining tasks and any final training settings still require documentation.',
        'Section 5 summarizes realized updates for all completed fits in this snapshot. The accompanying training table retains missing fits and links each completed fit to its hashed learning history.')
    start='The interface has been exercised with fictional prices across the planned task shapes, including restoration in a separate process.'
    end='A separate reporting utility checks the frozen training configuration'
    a,b=text.split(start,1);_,c=b.split(end,1)
    inference=(
        'We checked the interface with fictional prices across the planned task shapes. A historical full-array audit '
        'also restored 240 neural predictors in 60 fresh CPU processes for the 12 complete tasks. All 179,288 '
        'origin-model prediction paths reproduced their saved values exactly with the original prediction batches '
        'of 512. That audit opened no saved target arrays.\n\n'
        'The separate twenty-task package retains all eight models for each task and selects the latest completed '
        'chronological fold, without using model scores to choose a fold. Its registry records a 71-fold snapshot '
        'and marks the study as partial. A fresh-process check of one original origin per task covers 160 model '
        'forecast calls. The checks use the interface’s fixed numerical tolerances and retain measured differences; '
        'they do not claim full-array equality for all twenty tasks. A saved minute H60-return command-line example '
        'uses 480 past input bars and reproduces its checked API output exactly. The package contains model '
        'parameters and metadata, while the caller supplies historical inputs.\n\n'
        'A further CPU check started from immutable real-data mid-epoch GRU and xPatch checkpoints. For each model, '
        'we compared two updates in one process with one update, deliberate process exit, and one update in a fresh '
        'process. All checkpoint fields matched, including optimizer, batch cursor, permutation and random states. '
        f'The six processes took {restart["elapsed_seconds"]:.1f} seconds and performed eight updates across the '
        'compared trajectories. This bounded engineering check changed no original worker or experiment result '
        'and scored no validation targets. It checks two real CPU checkpoint states; full-fold GPU recovery uses '
        'separate synthetic fixtures.\n\n')
    text=a+inference+end+c
    snapshot=(f'**Historical development snapshot:** as of 8 November 2026 at {stamp} Beijing time, '
        f'{complete} of 20 tasks have all five development folds and {n} of 100 planned task-folds are complete. '
        'All twenty tasks have at least one completed fold. The minute matrix and independent confirmation remain '
        'incomplete. The optional shared-versus-separate comparison has not been run. Percentages below compare '
        'endpoint MSE with matched persistence or zero return; they are not investment returns.')
    text=section(text,'## 5. Experimental results','### Development coverage and baseline magnitudes',snapshot)
    old_header='| Task | Price folds | Return folds |'
    pos=text.index(old_header);endpos=text.index('\n\n',pos)
    coverage_rows=[]
    for scale in SCALES:
        for horizon in sorted({t['horizon'] for t in tasks if t['profile']==scale}):
            pair=[next(t for t in tasks if (t['profile'],t['horizon'],t['target'])==(scale,horizon,target))
                  for target in ('future_close','future_log_return')]
            coverage_rows.append([task_name(pair[0])]+[f"{len(t['completed_folds'])}/5" for t in pair])
    text=text[:pos]+table(['Task','Price folds','Return folds'],coverage_rows)+text[endpos:]
    with (source/'TRAINING_MODELS.csv').open(newline='',encoding='utf8') as f:
        trained=[r for r in csv.DictReader(f) if r['status']=='completed']
    training_rows=[]
    for scale in SCALES+('total',):
        subset=trained if scale=='total' else [r for r in trained if r['profile']==scale]
        training_rows.append([scale.title(),len(subset),sum(r['epoch_cap_reached']=='True' for r in subset),
                              sum(r['stopped_early']=='True' for r in subset),number(statistics.median(int(r['steps']) for r in subset))])
    timing=result['elapsed_accounting']
    budget=(f'This table covers {len(trained)} completed neural model-fold fits across the {n} completed task-folds, '
        'including partial minute tasks. The current completed fits share a 20-epoch cap. The cap and early-stop '
        'flag can overlap at the last allowed epoch. Counts exclude post-hoc anchoring and split-neutral runs.\n\n'+
        table(['Scale','Neural fits','At 20 epochs','Early-stop flag','Median updates'],training_rows)+'\n\n'+
        'Reaching the cap or an early-stop condition does not establish convergence. The training CSV records '
        'realized epochs, best epochs, updates and original fit/stop sample counts.\n\n'+
        'The researcher requested low load until 03:30 Beijing on 8 November. The controller resumed the six '
        'original workers at 03:30:01 with their original process identities and CPU affinities, preserving '
        'remaining queue budgets. Some raw monotonic elapsed receipts include the requested suspension despite '
        'their original active-time label. We preserve those receipts and use an external pause-envelope estimate '
        f'of {recovery["pause_envelope_seconds"]:.1f} seconds for the affected stages. At this snapshot, the sum '
        f'of {timing["unique_completed_stage_records"]} unique completed-stage wall envelopes is '
        f'{timing["sum_recorded_wall_envelopes_seconds"]/3600:.3f} hours; subtracting known suspension envelopes '
        f'gives approximately {timing["sum_suspension_adjusted_elapsed_estimates_seconds"]/3600:.3f} stage-hours. '
        'Parallel stages overlap, and unfinished work is excluded. These sums are neither the study’s wall-clock '
        'duration nor measured per-model CPU time. Queue deadline accounting remains separate.')
    text=section(text,'### Realized training budget','### Additional minute tasks: first-fold snapshot',budget)
    minute_rows=[]
    for t in [t for t in tasks if t['profile']=='minute']:
        minute_rows.append([f"H{t['horizon']} "+('price' if t['target']=='future_close' else 'return'),
                            ', '.join(map(str,t['completed_folds'])),f"{t['pooled']['origins']:,}"]+
                           [percent(t['pooled']['models'][m]['relative_improvement']) for m in MODELS[1:]])
    minute=(f'We recomputed all {minute_n} completed minute folds. Each row pools only the listed folds; '
        'different fold sets cover different market periods, so the rows do not supply a matched comparison '
        'between horizons or target types. Missing folds remain explicit in the coverage table and CSV.\n\n'+
        table(['Task','Folds','Origins']+list(NAMES[1:]),minute_rows)+'\n\n'+
        f'For H60 returns, folds {", ".join(map(str,h60["completed_folds"]))} contribute '
        f'{h60["pooled"]["origins"]:,} origins. Missing folds: '+', '.join(map(str,h60['missing_folds']))+'. '
        'The table below retains fold-specific endpoint comparisons; the earlier sampled-origin pilot is excluded.\n\n')
    frows=[]
    for fold in range(1,6):
        f=next((f for f in h60['folds'] if f['fold']==fold),None)
        frows.append([fold,f['origins'] if f else 'pending',number(f['models']['last_close']['endpoint_mse']) if f else '—']+
                     [percent(f['models'][m]['relative_improvement']) if f else '—' for m in NEURAL])
    minute+=table(['H60 return fold','Origins','Zero-return MSE','MLP','GRU','LSTM','xPatch'],frows)+'\n\n'+(
        'The annual short-history task has 47 overlapping validation origins across its five folds. For annual '
        'returns, GRU improvement changes from -234.36% in fold 1 to 88.74% in fold 2. The small number of market '
        'periods limits inference, and the return target excludes cash dividends.')
    text=section(text,'### Additional minute tasks: first-fold snapshot','### Annual sensitivity: nonoverlapping forecast windows',minute)
    text=replace_once(text,'### Additional minute tasks: first-fold snapshot','### Minute-scale results at the recorded snapshot')
    old=text.split('### Descriptive uncertainty across completed tasks\n\n',1)[1].split('\n\nFor example,',1)[0]
    uncertainty=(
        'We summarize paired endpoint squared-loss differences for all twenty tasks using 5,000 non-circular '
        'moving-block resamples, seed 2026 and nominal 95% percentile intervals. The full source session grid '
        'retains gaps between observed validation groups. The supplement contains 320 task–model–block rows, '
        f'including {result["interval_status_counts"].get("descriptive_development_only",0)} varying descriptive '
        f'intervals and {result["interval_status_counts"].get("degenerate_bootstrap",0)} degenerate cases with '
        'undefined bounds. It records each task’s completed fold set, including partial minute tasks. Negative '
        'model-minus-baseline loss differences favor the model. We report no development p-values. Prior score '
        'exposure, dependence and nonstationarity limit interpretation.')
    text=replace_once(text,old,uncertainty)
    text=replace_once(text,
        '**[Complete the conclusion after the remaining comparisons and any independent evaluation.]**',
        'Under the recorded recipes, persistence remains a strong reference and target construction materially '
        'affects apparent improvement. Completing the outstanding minute folds is necessary before drawing a '
        'full twenty-task comparison; independent confirmation requires a future protocol and eligible data.')
    availability=(
        'The version-controlled repository contains task definitions, source and split checks, independent '
        'scoring tools, the historical predictor interface, the all-task results compiler and the bounded '
        'checkpoint restart check. The code release receipt records the Git commit and remote verification. '
        'The source manifest identifies the exact Alpaca daily and minute versions and derived bar files. '
        'The researcher-confirmed permission record covers Alpaca and Kibot; the experiments reported here use '
        'Alpaca. The local supplement includes all 160 pooled model rows, 800 model-fold reporting cells, '
        '2,544 model/step rows, 320 interval rows and 400 planned neural-fit records. The package registry '
        'and its audit describe the available twenty-task predictors and the missing experimental folds. '
        'Matching licensed source files and artifacts are required to reproduce the local numerical evidence. '
        'No archive DOI has been assigned to this working snapshot.')
    text=section(text,'## 7. Data and code availability','## References',availability)
    appendix=(
        '**Historical development evidence at the recorded snapshot.** The following twenty tables retain '
        'the completed fold sets and all eight models. Twelve tasks have five folds; the minute rows use '
        'partial histories and cannot establish the complete comparison.\n\n'
        'We pool endpoint MSE, RMSE, MAE and path errors by origin count. Price MSE uses (USD per share)²; '
        'price RMSE/MAE use USD per share. Return MSE uses squared log-return units. Improvement is '
        '100 × (1 − model endpoint MSE / matched persistence MSE). Direction accuracy covers nonzero predicted '
        'endpoint changes; an all-abstaining predictor has no accuracy value. The CSV supplement retains '
        'fold-specific errors, path MAE and every prediction step. Dates below are recorded New York completion '
        'timestamps and do not establish point-in-time vendor availability.\n\n')
    for t in tasks:
        heading=task_name(t)+(': nominal close' if t['target']=='future_close' else ': cumulative log return')
        folds=t['folds'];p=t['pooled']
        appendix+=f"### {heading}\n\nCompleted folds: {', '.join(map(str,t['completed_folds']))}; "
        appendix+=f"missing folds: {', '.join(map(str,t['missing_folds'])) or 'none'}; {p['origins']:,} origins. "
        appendix+=f"First and last decisions: {folds[0]['first_decision_ny']} to {folds[-1]['last_decision_ny']}; "
        appendix+=f"last target completion: {max(f['last_completion_ny'] for f in folds)}.\n\n"
        appendix+=table(['Model','End MSE','End RMSE','End MAE','Path MSE','Improvement','Direction coverage','Active accuracy'],
            [[name]+[number(p['models'][model][key]) for key in ('endpoint_mse','endpoint_rmse','endpoint_mae','path_mse')]+
             [percent(p['models'][model][key]) for key in ('relative_improvement','direction_coverage','direction_accuracy_when_active')]
             for model,name in zip(MODELS,NAMES)])+'\n\n'
    text=text.split('## Appendix A. Detailed historical development results',1)[0]+'## Appendix A. Detailed historical development results\n\n'+appendix
    require(re.findall(r'\\\[.*?\\\]|\\\(.*?\\\)',original,re.S)==re.findall(r'\\\[.*?\\\]|\\\(.*?\\\)',text,re.S),'Equation changed')
    require(original.split('## References\n',1)[1].split('## Appendix A.',1)[0]==text.split('## References\n',1)[1].split('## Appendix A.',1)[0],'Bibliography changed')
    directory=OUT/'private_working_manuscripts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');directory.mkdir()
    (directory/'PRIVATE_WORKING_MANUSCRIPT_EN.md').write_text(text,encoding='utf8')
    for folder in ('figures','paper_artifacts','reference_evidence_20261108'):shutil.copytree(PARENT/folder,directory/folder)
    for name in ('DATA_PERMISSION_CONFIRMATION.json','DATA_PROVENANCE_MANIFEST.json','RESTORATION_AUDIT.json',
                 'PRIVATE_ANNUAL_NONOVERLAP_RESULTS.csv'):shutil.copy2(PARENT/name,directory/name)
    shutil.copytree(source,directory/'all_task_results')
    for path,name in ((RESTART,'REAL_TRAINING_RESTART_AUDIT.json'),(PACKAGE_AUDIT,'STUDY_PACKAGE_AUDIT.json'),
                      (NIGHT,'ACTUAL_NIGHT_RESUME_VERIFICATION.json'),(Path(protocol['coverage']),'COVERAGE_SNAPSHOT.json')):
        shutil.copy2(path,directory/name)
    audit={'status':'all_twenty_task_interim_results_and_engineering_evidence_integrated',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'parent_manuscript_sha256':sha(PARENT/'PRIVATE_WORKING_MANUSCRIPT_EN.md'),
        'manuscript_sha256':sha(directory/'PRIVATE_WORKING_MANUSCRIPT_EN.md'),'results_sha256':sha(source/'RESULTS.json'),
        'restart_audit_sha256':sha(RESTART),'package_audit_sha256':sha(PACKAGE_AUDIT),
        'recovery_evidence_sha256':sha(NIGHT),'script_sha256':sha(Path(__file__)),
        'completed_folds':n,'complete_tasks':complete,'task_tables':20,'equations_unchanged':True,
        'bibliography_unchanged':True,'independent_test':False,'Q3_values_read':False,
        'limitations':['Minute folds remain incomplete.','The package uses its own recorded earlier 71-fold snapshot.']}
    with (directory/'MANUSCRIPT_UPDATE_AUDIT.json').open('x') as f:json.dump(audit,f,indent=2);f.write('\n')
    print(json.dumps({'status':audit['status'],'directory':str(directory),'completed_folds':n,'sha256':audit['manuscript_sha256']}))


if __name__=='__main__':main()
