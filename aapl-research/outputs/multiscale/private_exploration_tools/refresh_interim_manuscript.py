"""Refresh all result sections from one verified, explicitly partial snapshot."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import statistics
from zoneinfo import ZoneInfo

from update_all20_working_manuscript import (MODELS, NAMES, NEURAL, SCALES, load, number, percent,
                                             require, section, sha, table, task_name)

ROOT = Path('/home/ayaka/wslcode/aapl-research')
OUT = ROOT / 'outputs/multiscale'
PARENT = OUT / 'private_working_manuscripts/20261107T204817634573Z'
SPLIT_AUDIT = OUT / 'historical_split_group_audits/20261108_original_monthly_all_metrics'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    result_dir = args.results.resolve()
    result = load(result_dir / 'RESULTS.json')
    protocol = load(result_dir / 'COMPILATION_PROTOCOL.json')
    require(sha(result_dir / 'COMPILATION_PROTOCOL.json') == result['protocol_sha256'], 'Result plan changed')
    for name, identity in result['csv_sha256'].items():
        require(sha(result_dir / name) == identity, 'Result CSV changed')
    coverage_path = Path(protocol['coverage'])
    require(sha(coverage_path) == protocol['coverage_sha256'], 'Result coverage changed')
    coverage = load(coverage_path)
    split = load(SPLIT_AUDIT / 'SPLIT_GROUP_AUDIT.json')
    require(split['all_prior_group_values_match'] is True and
            split['weighted_group_errors_match_pooled_errors'] is True and
            split['group_model_rows'] == 32 and split['new_training_started'] is False,
            'Monthly split-group audit differs')
    n, complete = result['completed_folds'], result['complete_tasks']
    require(74 < n < 100 and len(result['tasks']) == 20 and result['p_values_reported'] is False and
            result['Q3_values_read'] is False and coverage['completed_folds'] == n,
            'Expected a later partial development snapshot')
    stamp = datetime.fromisoformat(coverage['created_at_utc']).astimezone(ZoneInfo('Asia/Shanghai')).strftime('%H:%M')
    tasks = sorted(result['tasks'], key=lambda t: (SCALES.index(t['profile']), t['horizon'], t['target']))
    source = PARENT / 'PRIVATE_WORKING_MANUSCRIPT_EN.md'
    text = original = source.read_text(encoding='utf8')
    text = text.replace(text.splitlines()[2],
        f'**WORKING MANUSCRIPT: EXPERIMENTS INCOMPLETE.** This revision reports {n}/100 completed historical '
        'development folds. Permission for all Alpaca and Kibot data follows the researcher confirmation. '
        'Missing minute folds, the separate package snapshot and the original compute limits remain explicit.', 1)
    h60 = next(t for t in tasks if (t['profile'], t['horizon'], t['target']) == ('minute', 60, 'future_log_return'))
    finished_prices = [t for t in tasks if t['target'] == 'future_close' and len(t['completed_folds']) == 5]
    wins = sum(t['pooled']['models'][m]['relative_improvement'] > 0 for t in finished_prices for m in NEURAL)
    g = h60['pooled']['models']['gru']['relative_improvement']
    abstract = (
        'Forecast accuracy for one stock can vary across horizons and target representations. We compared '
        'persistence, drift, seasonal repetition, ridge regression, an MLP, GRU, LSTM and an adapted xPatch '
        'model on Apple Inc. (AAPL) prices and cumulative log returns across five temporal scales. '
        f'This historical development snapshot contains {n} of 100 planned folds, with {complete} of 20 tasks '
        f'complete. Among {len(finished_prices) * 4} neural task–model comparisons on complete nominal-price '
        f'tasks, {wins} had lower pooled endpoint MSE than persistence. The minute H60 return GRU had '
        f'{abs(100 * g):.2f}% {"lower" if g > 0 else "higher"} pooled endpoint MSE than zero return across '
        f'{len(h60["completed_folds"])} available folds. In a post-hoc split-neutral monthly comparison, the '
        'GRU’s relative MSE improvements were 3.96%, -24.67% and 3.99% across three seeds. The annual task '
        'had 47 overlapping validation origins. We provide descriptive uncertainty tables, audited original '
        'fold designs and a historical prediction interface. Prior exposure, limited annual history and '
        'unfinished minute comparisons restrict the conclusions; the study supplies no independent '
        'confirmation or trading evaluation.')
    text = section(text, '## Abstract', '## 1. Introduction', abstract)
    substitutions = {
        'For local exploratory work, the user accepted approximately ten years of daily history. The annual variant uses a twelve-month lookback, which also differs from the original design.':
        'The exploratory annual variant used approximately ten years of daily history and a twelve-month lookback, departing from the original minimum history and lookback.',
        'Only complete full-origin folds are counted. The user accepted approximately ten years of daily history; the short-history annual configuration uses a twelve-month lookback and differs from the original design.':
        'Only complete full-origin folds are counted. The annual short-history configuration used approximately ten years of daily history and a twelve-month lookback, departing from the original design.',
        'and all choices remain subject to a final protocol.':
        'and the original execution protocols retain these historical settings.',
        'The proposed design uses five expanding chronological development folds for each task when the admitted history supports them.':
        'The historical design used five expanding chronological development folds for each task.',
        'The fold dates, task definitions, source identities, and finite configuration grid must be frozen before the first outer score is used.':
        'The original execution protocols and receipts retain task definitions, source identities, finite configurations and fold indices. These development records preserve prior score exposure; retrospective documentation does not make the evaluated periods independent.',
        'Paired endpoint loss differences are to be summarized by a non-circular moving-block procedure using the original session grid and origin-count weighting [15].':
        'Paired endpoint loss differences were summarized by a non-circular moving-block procedure using the original session grid and origin-count weighting [15].',
        'Proposed primary and sensitivity block lengths, expressed in session groups, are':
        'The primary and sensitivity block lengths, expressed in session groups, were',
        'the path errors proposed here.': 'the path errors evaluated here.',
        'our proposed single-stock comparison': 'our single-stock comparison',
        'the planned path errors.': 'the evaluated path errors.',
        'Our proposed chronological design': 'Our chronological design',
        'The planned targets use nominal prices.': 'The original targets used nominal prices.',
        'The proposed study concerns one stock, uses different source histories at different frequencies, and may have few nonoverlapping annual outcome windows, which can still be dependent.':
        'The study concerns one stock and uses different source histories at different frequencies. The annual sensitivity contains only four nonoverlapping outcome windows, which can still be dependent.',
    }
    edits = []
    for before, after in substitutions.items():
        if before in text:
            require(text.count(before) == 1, 'Ambiguous editorial sentence')
            text = text.replace(before, after, 1)
            edits.append({'before': before, 'after': after})
    old_split = next(line for line in text.splitlines() if line.startswith('The proposed sensitivity analysis groups'))
    new_split = (
        'After inspecting the historical development results, we classified a monthly H21 path as crossing '
        'the recorded split when its origin session preceded the effective date and its endpoint session '
        'was on or after that date. We recomputed group origin counts, endpoint MSE and MAE, full-path MSE '
        'and MAE, baseline squared-error contributions and relative endpoint-MSE improvements for all eight '
        'models. The subgroup metrics pooled back to the original errors, and all previously recorded '
        'subgroup comparisons agreed. Only one recorded split event was covered. These overlapping '
        'windows form a retrospective descriptive breakdown, not independent corporate-action events, a '
        'model-selection rule or a confirmation test.')
    text = text.replace(old_split, new_split, 1)
    edits.append({'before': old_split, 'after': new_split})
    text = section(text, '## 5. Experimental results', '### Development coverage and baseline magnitudes',
        f'**Historical development snapshot:** as of 8 November 2026 at {stamp} Beijing time, {complete} of '
        f'20 tasks had all five development folds and {n} of 100 planned task-folds were complete. All twenty '
        'tasks had at least one completed fold. The minute matrix remained incomplete under the original '
        'compute limits. No shared-versus-separate experiment was run. Percentages below compare endpoint '
        'MSE with matched persistence or zero return; they are not investment returns.')
    pos = text.index('| Task | Price folds | Return folds |')
    end = text.index('\n\n', pos)
    rows = []
    for scale in SCALES:
        for horizon in sorted({t['horizon'] for t in tasks if t['profile'] == scale}):
            pair = [next(t for t in tasks if (t['profile'], t['horizon'], t['target']) ==
                         (scale, horizon, target)) for target in ('future_close', 'future_log_return')]
            rows.append([task_name(pair[0])] + [f"{len(t['completed_folds'])}/5" for t in pair])
    text = text[:pos] + table(['Task', 'Price folds', 'Return folds'], rows) + text[end:]
    with (result_dir / 'TRAINING_MODELS.csv').open(newline='') as stream:
        trained = [r for r in csv.DictReader(stream) if r['status'] == 'completed']
    rows = []
    for scale in SCALES + ('total',):
        subset = trained if scale == 'total' else [r for r in trained if r['profile'] == scale]
        rows.append([scale.title(), len(subset), sum(r['epoch_cap_reached'] == 'True' for r in subset),
                     sum(r['stopped_early'] == 'True' for r in subset),
                     number(statistics.median(int(r['steps']) for r in subset))])
    timing = result['elapsed_accounting']
    budget = (f'This table covers {len(trained)} completed neural model-fold fits across {n} completed '
        'task-folds. These fits shared a 20-epoch cap. The cap and early-stop flag can overlap at the last '
        'allowed epoch. Counts exclude post-hoc anchoring and split-neutral runs.\n\n' +
        table(['Scale', 'Neural fits', 'At 20 epochs', 'Early-stop flag', 'Median updates'], rows) + '\n\n'
        'Reaching a cap or early-stop condition does not establish convergence. The CSV retains realized '
        'epochs, best epochs, updates and original fit/stop counts.\n\n'
        'Following the requested low-load interval, six original workers resumed at 03:30:01 Beijing with '
        'their original process identities, affinities and remaining budgets. Some raw monotonic receipts '
        'include the 14122.1-second suspension despite their original active-time label. The original receipts '
        'are preserved; a separate estimate subtracts that known suspension envelope for affected stages. '
        f'At this snapshot, {timing["unique_completed_stage_records"]} unique completed-stage wall envelopes '
        f'sum to {timing["sum_recorded_wall_envelopes_seconds"] / 3600:.3f} hours; their suspension-adjusted '
        f'estimate is {timing["sum_suspension_adjusted_elapsed_estimates_seconds"] / 3600:.3f} stage-hours. '
        'Parallel stages overlap, and unfinished work is excluded. These sums are neither the study wall-clock '
        'duration nor measured per-model CPU time. Original queue deadline accounting remains separate.')
    text = section(text, '### Realized training budget', '### Minute-scale results at the recorded snapshot', budget)
    rows = []
    for t in [t for t in tasks if t['profile'] == 'minute']:
        rows.append([f"H{t['horizon']} " + ('price' if t['target'] == 'future_close' else 'return'),
                     ', '.join(map(str, t['completed_folds'])), f"{t['pooled']['origins']:,}"] +
                    [percent(t['pooled']['models'][m]['relative_improvement']) for m in MODELS[1:]])
    minute = (f'We recomputed all {n - 60} completed minute folds. Each row pools only the listed folds; '
        'different fold sets cover different market periods and do not supply a matched comparison between '
        'horizons or target types. Missing folds remain explicit.\n\n' +
        table(['Task', 'Folds', 'Origins'] + list(NAMES[1:]), rows) + '\n\n' +
        f"For H60 returns, folds {', '.join(map(str, h60['completed_folds']))} contribute "
        f"{h60['pooled']['origins']:,} origins. Missing folds: {', '.join(map(str, h60['missing_folds']))}. "
        'The following fold-specific endpoint comparisons exclude the sampled-origin pilot.\n\n')
    rows = []
    for number_ in range(1, 6):
        fold = next((f for f in h60['folds'] if f['fold'] == number_), None)
        rows.append([number_, fold['origins'] if fold else 'incomplete',
                     number(fold['models']['last_close']['endpoint_mse']) if fold else '—'] +
                    [percent(fold['models'][m]['relative_improvement']) if fold else '—' for m in NEURAL])
    minute += table(['H60 return fold', 'Origins', 'Zero-return MSE', 'MLP', 'GRU', 'LSTM', 'xPatch'], rows)
    minute += ('\n\nThe annual short-history task retained 47 overlapping validation origins. Its annual-return '
               'GRU improvement changed from -234.36% in fold 1 to 88.74% in fold 2. The small number of '
               'market periods limits inference; the return target excludes cash dividends.')
    text = section(text, '### Minute-scale results at the recorded snapshot',
                   '### Annual sensitivity: nonoverlapping forecast windows', minute)
    uncertainty = text.split('### Descriptive uncertainty across completed tasks\n\n', 1)[1].split('\n\nFor example,', 1)[0]
    new_uncertainty = (
        'We summarized paired endpoint squared-loss differences for all twenty tasks using 5,000 '
        'non-circular moving-block resamples, seed 2026 and nominal 95% percentile intervals. The full source '
        'session grid retained gaps between observed validation groups. The supplement contains 320 '
        f'task–model–block rows, including {result["interval_status_counts"].get("descriptive_development_only", 0)} '
        f'varying descriptive intervals and {result["interval_status_counts"].get("degenerate_bootstrap", 0)} '
        'degenerate cases with undefined bounds. Each task retains its completed fold set, including partial '
        'minute tasks. Negative model-minus-baseline loss differences favor the model. We report no '
        'development p-values. Prior exposure, dependence and nonstationarity limit interpretation.')
    text = text.replace(uncertainty, new_uncertainty, 1)
    split_tables = ('The following tables retain every model in both groups. Price and return tasks each '
        'have 21 crossing and 1,949 non-crossing origins. The supplement retains endpoint MAE, full-path '
        'MSE/MAE and every step as well as endpoint MSE. These are post-exposure comparisons around one '
        'split event.\n\n')
    for target, label in (('future_close', 'Nominal close'), ('future_log_return', 'Cumulative log return')):
        groups = split['groups'][target]
        rows = []
        for model, name in zip(MODELS, NAMES):
            crossing = groups['crosses_split']['models'][model]
            other = groups['no_split']['models'][model]
            rows.append([name, number(crossing['endpoint_mse']), number(other['endpoint_mse']),
                         percent(crossing['relative_improvement']), percent(other['relative_improvement'])])
        split_tables += label + ': split-group endpoint errors.\n\n'
        split_tables += table(['Model', 'Crossing MSE', 'Non-crossing MSE', 'Crossing improvement',
                               'Non-crossing improvement'], rows) + '\n\n'
    boundary = 'A subsequent label-only diagnostic'
    require(text.count(boundary) == 1, 'Split-result insertion boundary differs')
    text = text.replace(boundary, split_tables + boundary, 1)
    appendix = (
        '**Historical development evidence at the recorded snapshot.** All twenty tasks and all eight '
        f'models are retained below. {complete} tasks have five folds; partial minute histories remain '
        'explicit and do not establish a complete comparison.\n\n'
        'Endpoint MSE, RMSE, MAE and path errors are pooled by origin count. Price MSE uses (USD per share)²; '
        'price RMSE/MAE use USD per share. Return MSE uses squared log-return units. Improvement is '
        '100 × (1 − model endpoint MSE / matched persistence MSE). Direction accuracy covers nonzero '
        'predicted endpoint changes; an all-abstaining predictor has no accuracy value. The CSV retains '
        'fold-specific errors, path MAE and every prediction step. Dates are recorded New York bar completion '
        'timestamps and do not establish historical point-in-time vendor availability.\n\n')
    for task in tasks:
        p, folds = task['pooled'], task['folds']
        heading = task_name(task) + (': nominal close' if task['target'] == 'future_close' else ': cumulative log return')
        appendix += f"### {heading}\n\nCompleted folds: {', '.join(map(str, task['completed_folds']))}; "
        appendix += f"missing folds: {', '.join(map(str, task['missing_folds'])) or 'none'}; {p['origins']:,} origins. "
        appendix += f"First and last decisions: {folds[0]['first_decision_ny']} to {folds[-1]['last_decision_ny']}; "
        appendix += f"last target completion: {folds[-1]['last_completion_ny']}.\n\n"
        appendix += table(['Model', 'End MSE', 'End RMSE', 'End MAE', 'Path MSE', 'Improvement',
                           'Direction coverage', 'Active accuracy'],
            [[name] + [number(p['models'][m][k]) for k in ('endpoint_mse', 'endpoint_rmse', 'endpoint_mae', 'path_mse')] +
             [percent(p['models'][m][k]) for k in ('relative_improvement', 'direction_coverage', 'direction_accuracy_when_active')]
             for m, name in zip(MODELS, NAMES)]) + '\n\n'
    text = section(text, '## Appendix A. Detailed historical development results',
                   '## Appendix B. Forecast timing and original fold design', appendix)
    text = text.replace('Numerical results in Section 5 and Appendix A remain at the separately recorded 74-fold snapshot.',
                        f'Numerical results in Section 5 and Appendix A use the separately recorded {n}-fold snapshot.')
    require(re.findall(r'\\\[.*?\\\]|\\\(.*?\\\)', original, re.S) ==
            re.findall(r'\\\[.*?\\\]|\\\(.*?\\\)', text, re.S), 'Equation changed')
    require(original.split('## References\n', 1)[1].split('## Appendix A.', 1)[0] ==
            text.split('## References\n', 1)[1].split('## Appendix A.', 1)[0], 'Bibliography changed')
    destination = OUT / 'private_working_manuscripts' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination.mkdir()
    for folder in ('figures', 'paper_artifacts', 'reference_evidence_20261108', 'forecast_timing'):
        shutil.copytree(PARENT / folder, destination / folder)
    shutil.copytree(result_dir, destination / 'all_task_results')
    shutil.copytree(SPLIT_AUDIT, destination / 'monthly_split_groups')
    analysis_sources = destination / 'analysis_sources'
    analysis_sources.mkdir()
    for name in ('audit_monthly_split_groups.py', 'refresh_interim_manuscript.py',
                 'integrate_forecast_timing.py', 'verify_timing_manuscript.py'):
        shutil.copy2(OUT / 'private_exploration_tools' / name, analysis_sources / name)
    for name in ('DATA_PERMISSION_CONFIRMATION.json', 'DATA_PROVENANCE_MANIFEST.json',
                 'REAL_TRAINING_RESTART_AUDIT.json', 'STUDY_PACKAGE_AUDIT.json',
                 'ACTUAL_NIGHT_RESUME_VERIFICATION.json', 'COMBINED_81_FOLD_RESTORATION.json',
                 'FINAL_EDITORIAL_NOTES.md', 'REMAINING_REQUIREMENTS.md'):
        if (PARENT / name).exists(): shutil.copy2(PARENT / name, destination / name)
    manuscript = destination / source.name
    manuscript.write_text(text, encoding='utf8')
    insertion = load(PARENT / 'TIMING_INSERTION_AUDIT.json')
    insertion.update({'parent': str(PARENT), 'parent_markdown_sha256': sha(source),
                      'markdown_sha256': sha(manuscript), 'numerical_result_snapshot_completed_folds': n,
                      'prior_content_preserved': False, 'note': 'Explicit numerical and editorial refresh; see SNAPSHOT_REFRESH_AUDIT.json.'})
    (destination / 'TIMING_INSERTION_AUDIT.json').write_text(json.dumps(insertion, indent=2) + '\n')
    audit = {'status': 'all_twenty_task_partial_snapshot_and_editorial_refresh',
             'created_at_utc': datetime.now(timezone.utc).isoformat(), 'parent': str(PARENT),
             'parent_markdown_sha256': sha(source), 'manuscript_sha256': sha(manuscript),
             'results_sha256': sha(result_dir / 'RESULTS.json'), 'completed_folds': n,
             'complete_tasks': complete, 'editorial_substitutions': edits, 'equations_unchanged': True,
             'split_group_audit_sha256': sha(SPLIT_AUDIT / 'SPLIT_GROUP_AUDIT.json'),
             'split_group_tables': 2,
             'references_unchanged': True, 'script_sha256': sha(Path(__file__)), 'Q3_values_read': False}
    (destination / 'SNAPSHOT_REFRESH_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'destination': str(destination), 'completed_folds': n, 'complete_tasks': complete,
                      'editorial_substitutions': len(edits)}))


if __name__ == '__main__': main()
