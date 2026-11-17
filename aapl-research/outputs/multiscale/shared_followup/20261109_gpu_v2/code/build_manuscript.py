"""Gated shared-horizon addendum; never trains, loads market arrays, or overwrites prior papers.

Only --self-test is intended before the completed matrix and independent
statistics-review receipt exist. Formal output is manuscript/new_version/.
"""
from pathlib import Path
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[3]
V3B = REPO/'outputs/multiscale/mechanism_repair/20261109_v3b'
TITLE = 'Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales'
HORIZONS = (5, 15, 30, 60)
SEEDS = (2026, 2027, 2028)
CANDIDATE = 'tied_selected_shrink'
METHODS = tuple(a+'_'+m for a in ('tied', 'untied')
                for m in ('last', 'last_shrink', 'selected', 'selected_shrink'))
SOURCE_IDENTITIES = {
    'MAIN_PAPER_EN.md': '3ad38928486e703bbec1b494155ecab8f0b722ecb01cc3fb9f8186834590ae99',
    'SUPPLEMENT_EN.md': 'd003f6676d0e70576f5e3e6c0e18074a3775a56c31a9f97f63f3bda2902aa398',
    'AUTHOR_INFORMATION.json': '5ca14672c0f418f630534aebe27932618466e1bba422dd6039cc4784a3d578ff'}
AUTHORIZED_DEADLINE = 1794216709.0
DEADLINE = None
STOP = False


def guard():
    if STOP or (DEADLINE is not None and time.time() >= DEADLINE):
        raise InterruptedError('Manuscript interrupted or authorized deadline reached')


def interrupted(*_):
    global STOP
    STOP = True


def authorize_formal_deadline():
    global DEADLINE
    DEADLINE = AUTHORIZED_DEADLINE
    guard()
    authorization = read(BASE/'protocol/AUTHORIZATION.json')
    rules = read(BASE/'protocol/RULES.json')
    assert float(authorization['deadline_epoch']) == float(rules['deadline_epoch']) == AUTHORIZED_DEADLINE
    assert authorization['authorized'] is True and authorization['paid_resources'] is False
    guard()


def command_run(arguments, **kwargs):
    guard()
    result = subprocess.run(arguments, check=True, **kwargs)
    guard()
    return result


def command_output(arguments, **kwargs):
    guard()
    result = subprocess.check_output(arguments, **kwargs)
    guard()
    return result


def text_write(path, text):
    guard(); Path(path).write_text(text); guard()


def sha(path):
    guard()
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            guard()
            h.update(block)
    return h.hexdigest()


def read(path):
    guard()
    return json.loads(Path(path).read_text())


def write(path, value):
    guard()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    guard()


def load_module(path, name):
    guard()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def completion_gate(base=BASE):
    """Require completion and independent review BEFORE opening STATS/RESULTS.

    Reads JSON identities and byte hashes only. stage_release then verifies the
    complete 82-file analysis closure; no np.load or torch import is needed.
    """
    required = ('protocol/RULES.json', 'protocol/EVALUATION_FREEZE.json',
        'audit/COMPLETE.json', 'analysis/ANALYSIS_COMPLETE.json',
        'review/INDEPENDENT_STATISTICS_REVIEW.json', 'review/independent_statistics.py')
    assert all((base/name).is_file() for name in required), 'Matrix, analysis and independent statistics review must all exist'
    rules = read(base/'protocol/RULES.json')
    assert rules['admitted'] is True and rules['fixed_epochs'] == 20 and len(rules['jobs']) == 30
    assert len({j['id'] for j in rules['jobs']}) == 30
    expected = {(a, f, s) for a in ('tied', 'untied') for f in range(1, 6) for s in SEEDS}
    assert {(j['arm'], j['fold'], j['seed']) for j in rules['jobs']} == expected
    done_paths = [base/'runs'/j['id']/'DONE.json' for j in rules['jobs']]
    assert all(path.is_file() for path in done_paths), 'All 30 DONE receipts are required before results can be read'
    rule_hash = sha(base/'protocol/RULES.json')
    worker_wall_seconds = worker_cpu_seconds = 0.
    parameter_counts = {}
    for job, path in zip(rules['jobs'], done_paths):
        done = read(path)
        assert done['status'] == 'completed' and done['job'] == job
        assert done['rules_sha256'] == rule_hash and done['fixed_epochs'] == 20
        assert done['Q3_values_read'] is False and done['independent_confirmation'] is False
        worker_wall_seconds += float(done.get('wall_seconds', 0.))
        worker_cpu_seconds += float(done.get('cpu_seconds', 0.))
        if 'parameters' in done:
            parameter_counts.setdefault(job['arm'], set()).add(done['parameters'])
    complete = read(base/'audit/COMPLETE.json')
    assert complete['status'] == 'completed' and complete['bundles'] == 30 and complete['Q3_values_read'] is False
    analysis = read(base/'analysis/ANALYSIS_COMPLETE.json')
    assert analysis['status'] == 'completed' and analysis['market_training_complete'] is True
    assert analysis['complete_bundles'] == 30 and analysis['all_five_folds_three_seeds'] is True
    assert analysis['rules_sha256'] == rule_hash
    review = read(base/'review/INDEPENDENT_STATISTICS_REVIEW.json')
    assert review['status'] == 'passed'
    for receipt in (analysis, review):
        assert receipt['Q3_values_read'] is False and receipt['independent_confirmation'] is False
        assert receipt['research_goal_achieved'] is False
    for field, name in (('rules_sha256', 'protocol/RULES.json'),
                        ('analysis_complete_sha256', 'analysis/ANALYSIS_COMPLETE.json'),
                        ('evaluation_freeze_sha256', 'protocol/EVALUATION_FREEZE.json'),
                        ('script_sha256', 'review/independent_statistics.py')):
        assert review[field] == sha(base/name), f'Independent-review identity mismatch: {field}'
    tasks = {f'minute_h{h}_future_log_return' for h in HORIZONS}
    assert set(review['tasks']) == tasks
    assert len(review['fold_checks']) == 20
    assert {(row['horizon'], row['fold']) for row in review['fold_checks']} == {
        (h, f) for h in HORIZONS for f in range(1, 6)}
    assert all(row['status'] == 'passed' for row in review['fold_checks'])
    # Completion + hash-bound independent review now permits result parsing.
    stage = load_module(BASE/'code/stage_release.py', 'manuscript_release_gate')
    stage.BASE = base
    stage.guard = guard
    stage.verify_complete_market(base/'analysis/ANALYSIS_COMPLETE.json')
    stats = read(base/'analysis/STATS.json')
    for task in tasks:
        actual, checked = stats['tasks'][task], review['tasks'][task]
        for key in ('horizon', 'mechanism', 'candidate_checks', 'candidate_control_directions',
                    'candidate_performance_gate_passed', 'primary_for_study'):
            assert actual[key] == checked[key], f'Reviewed decision mismatch: {task}/{key}'
        assert set(actual['summary']) == set(checked['summary'])
        for method, row in actual['summary'].items():
            assert math.isclose(row['endpoint_mse'], checked['summary'][method]['endpoint_mse'],
                                rel_tol=2e-9, abs_tol=2e-14)
    return stats, review, {'rules_sha256': rule_hash,
        'analysis_complete_sha256': sha(base/'analysis/ANALYSIS_COMPLETE.json'),
        'statistics_sha256': sha(base/'analysis/STATS.json'),
        'results_sha256': sha(base/'analysis/RESULTS.json'),
        'evaluation_freeze_sha256': sha(base/'protocol/EVALUATION_FREEZE.json'),
        'independent_statistics_review_sha256': sha(base/'review/INDEPENDENT_STATISTICS_REVIEW.json'),
        'recorded_training_compute': {'summed_worker_wall_seconds': worker_wall_seconds,
            'summed_worker_CPU_seconds': worker_cpu_seconds,
            'parameter_counts': {k: sorted(v) for k, v in parameter_counts.items()},
            'wall_time_meaning': 'sum of overlapping worker receipts, not queue elapsed or GPU kernel time'}}


def pct(value):
    return 'undefined' if value is None else f'{value:+.3f}%'


def table(headings, rows):
    return '\n'.join(['| '+' | '.join(headings)+' |', '| '+' | '.join(['---']*len(headings))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def interval_text(row):
    bounds = row.get('relative_ci95_pct')
    return row['status'] if bounds is None else f'[{bounds[0]:+.3f}, {bounds[1]:+.3f}]%'


def exact_replace(text, old, new):
    assert text.count(old) == 1, f'Expected one preceding-paper anchor: {old}'
    return text.replace(old, new, 1)


def check_sources():
    for name, expected in SOURCE_IDENTITIES.items():
        assert sha(V3B/'manuscript'/name) == expected, f'Preceding manuscript identity changed: {name}'
    author = read(V3B/'manuscript/AUTHOR_INFORMATION.json')
    assert author['authors'][0]['name_en'] == 'Wang Yang' and author['authors'][0]['email'] is None
    affiliation = author['affiliations'][0]
    assert affiliation['school'] == 'High School Affiliated to Shanghai Jiao Tong University'
    assert (affiliation['city'], affiliation['postal_code'], affiliation['country']) == ('Shanghai', '200439', 'China')


def compose(stats, compute=None):
    """Pure text transformation; accepts only the supplied statistics object."""
    guard(); check_sources()
    assert stats['candidate'] == CANDIDATE and stats['primary_task'] == 'minute_h60_future_log_return'
    assert stats['Q3_values_read'] is False and stats['independent_confirmation'] is False
    assert stats['research_goal_achieved'] is False
    entries = [stats['tasks'][f'minute_h{h}_future_log_return'] for h in HORIZONS]
    primary = entries[-1]
    mechanism = primary['mechanism']['passed']; candidate = primary['candidate_performance_gate_passed']
    assert isinstance(mechanism, bool) and isinstance(candidate, bool)
    assert mechanism is stats['primary_mechanism_gate_passed']
    assert candidate is stats['primary_candidate_performance_gate_passed']
    gain = primary['summary'][CANDIDATE]['improvement_pct_over_zero']
    mechanism_gain = 100*(1-primary['summary']['tied_last']['endpoint_mse']/primary['summary']['untied_last']['endpoint_mse']) if primary['summary']['untied_last']['endpoint_mse'] > 0 else None
    mechanism_sentence = ('The matched raw final-checkpoint sharing contrast passes its predefined mechanism gate.' if mechanism else
                          'The matched raw final-checkpoint sharing contrast fails its predefined mechanism gate.')
    candidate_sentence = ('The sole candidate passes the predefined minute-H60 development performance gate.' if candidate else
                          'The sole candidate fails the predefined minute-H60 development performance gate.')
    interpretation = ('The tested implementation supports a local sharing benefit together with a qualifying candidate on exposed minute-H60 development data.' if mechanism and candidate else
        'The tested implementation supports a local sharing benefit, while the selected candidate still fails the required forecasting criteria.' if mechanism else
        'The candidate qualifies on the fixed development criteria, but the raw sharing contrast does not establish encoder sharing as the cause of that gain.' if candidate else
        'Neither a repeatable raw sharing benefit nor a qualifying forecasting candidate is established by this fixed study.')
    main = (V3B/'manuscript/MAIN_PAPER_EN.md').read_text()
    supplement = (V3B/'manuscript/SUPPLEMENT_EN.md').read_text()
    supplement = exact_replace(supplement,
        'Wang Yang. Post-observation supplement v3a; exposed historical development, 2026Q3 unscored.',
        'Wang Yang. Cumulative post-observation supplement: preceding v3a/v3b stages and the completed shared-horizon stage; exposed historical development, 2026Q3 unscored.')
    supplement = exact_replace(supplement,
        'The original absolute compute deadline is epoch 1794205909.6228158 (2026-11-09 14:31:49.623 Beijing). At most eight training processes use two logical CPUs each; analysis processes share CPUs 24 and 26. No GPU, payment, additional download or protected-period read occurs.',
        'For the preceding completed CPU stages described in S1–S14, the original deadline was epoch 1794205909.6228158 (2026-11-09 14:31:49.623 Beijing), with at most eight two-core training processes and shared analysis CPUs 24 and 26. Their no-GPU statement applies to those earlier stages only. The current separate shared-horizon stage in S15–S18 uses the subsequently authorized local RTX 5070 Ti Laptop GPU and extended absolute deadline of 2026-11-09 17:31:49 Beijing. No paid resources or protected-period reads are authorized in either stage.')
    original_table = main.split('**Table 1.**', 1)[1].split('**Table 2.**', 1)[0]
    abstract_new = f'A subsequent five-fold, three-seed, two-arm experiment trains four genuine horizon-specific decoders with a shared or independent nonlinear encoder. Its minute-H60 candidate endpoint-MSE improvement over zero is {pct(gain)}; the sharing-mechanism and candidate-performance gates {"pass" if mechanism else "fail"} and {"pass" if candidate else "fail"}, respectively. '
    main = exact_replace(main, 'We identify implementation effects and report failed predictive criteria. We have not established a forecasting method superior to the strong controls.',
        abstract_new+'We retain all preceding negative results and distinguish development performance from independent confirmation.')
    main = exact_replace(main, 'A separate shared-versus-independent-horizon market experiment remains unrun.',
        'A separate matched shared-versus-independent-horizon experiment now tests nonlinear encoder sharing across four genuine minute-return decoders; it retains the original five-scale comparison.')
    method = '''### 3.5 Matched four-head shared-versus-independent experiment

A post-observation development study compares shared and independent nonlinear encoders for minute H5/H15/H30/H60 returns. Both arms use identical H60-legal origins, 480-bar histories, price anchors, fit-only scalers and zero-return initialization. Four genuine native H-specific decoders, trend branches and anchor heads remain; shorter forecasts are not H60 prefixes. Parameter tying changes gradients and capacity (472,549 versus 524,797 parameters), so this tests the complete implementation intervention.

Thirty bundles cover five folds, three seeds and two arms, using matched Adam batches and twenty epochs. The loss sums four standardized mean path MSEs. One bundle checkpoint minimizes mean inner-stop endpoint MSE across heads over epochs 1–20, taking the earliest tie. Each head estimates alpha=clip(pᵀy/pᵀp,0,1) from inner-stop endpoint predictions and observed returns, then applies it to the whole path. The only predictive candidate is tied-selected-shrink; raw final tied versus untied H60 is the separate mechanism contrast.

The primary mechanism gate requires three positive seeds, at least four positive folds and supporting one-/five-day paired block intervals. The separate candidate gate requires three positive seeds, at least four positive folds and at least 1% gain against zero, plus supporting intervals against every registered control at both lengths. H60 has forty control entries including known duplicates; other horizons have seven and are supporting. Decisions use mean seed losses, not averaged forecasts. Five thousand bootstrap draws yield nominal development intervals. Deterministic local-GPU FP32, fixed fused Adam, disabled TF32/mixed precision and graph/eager recovery undergo separate synthetic acceptance. Complete-matrix evaluation and independent statistical reconstruction precede manuscript integration; these do not establish independent market confirmation. Full controls and numerical details appear in S15–S18.

'''
    main = exact_replace(main, '## 4. Results\n', method+'## 4. Results\n')
    main_rows = [[f'H{h}', pct(entry['summary'][CANDIDATE]['improvement_pct_over_zero']),
        f"{entry['summary'][CANDIDATE]['positive_folds_over_zero']}/5",
        'pass' if entry['mechanism']['passed'] else 'fail',
        'pass' if entry['candidate_performance_gate_passed'] else 'fail'] for h, entry in zip(HORIZONS, entries)]
    compute_text = '' if compute is None else (f"Recorded training receipts sum to {compute['summed_worker_CPU_seconds']:.1f} CPU seconds and "
        f"{compute['summed_worker_wall_seconds']:.1f} worker elapsed seconds; the latter includes concurrent workers and is not queue elapsed time. ")
    results_text = f'''### 4.6 Completed matched horizon-sharing study

The complete thirty-bundle matrix is reported in Table 3. At the primary minute-H60 endpoint, raw final tied versus untied gains {pct(mechanism_gain)}. {mechanism_sentence} The sole tied-selected-shrink candidate gains {pct(gain)} against zero, with {primary['summary'][CANDIDATE]['positive_folds_over_zero']}/5 positive folds. {candidate_sentence} {interpretation}

**Table 3.** Sole four-head candidate: endpoint-MSE gain against zero from mean seed losses; positive folds compare the candidate with zero. H60 is primary; other horizons are supporting. Mechanism and candidate decisions are separate. All arm/checkpoint forms and intervals appear in S17.

{table(['H', 'Candidate gain', 'Folds', 'Mechanism', 'Candidate'], main_rows)}

{compute_text}No alternative checkpoint, head, scalar or ensemble is promoted from outer scores. The original hourly-H8 criterion is not retested by this minute experiment. Independent statistical reconstruction verifies the numerical results on exposed history; protected 2026Q3 remains sealed and eligible independent market confirmation remains outstanding.

'''
    main = exact_replace(main, '## 5. Limitations and conclusion\n', results_text+'## 5. Limitations and conclusion\n')
    main = exact_replace(main, 'No new recipe meets the designated stable predictive-improvement objective.',
        'None of the preceding initialization, normalization, small-model or context recipes meets the designated stable predictive-improvement objective. '+candidate_sentence)
    main = exact_replace(main, 'Shared-versus-independent horizon learning and eligible independent confirmation remain unfinished.',
        'The matched minute shared-versus-independent experiment is complete; its local result does not complete the original hourly return objective, and eligible independent confirmation remains unfinished.')
    main = exact_replace(main, 'The local context addendum contains the updated manuscripts and forty-model input-only inference/statistics replay.',
        'The current shared-horizon addendum adds the complete thirty-bundle development matrix, four-head input-only replay and independently checked statistics. The preceding local context addendum contains its forty-model input-only inference/statistics replay.')
    assert main.split('**Table 1.**', 1)[1].split('**Table 2.**', 1)[0] == original_table
    supplement = exact_replace(supplement, 'We have not run the separate shared-versus-independent horizon market experiment.',
        'The separate matched four-head shared-versus-independent horizon market experiment is now complete and reported in S15–S18 below.')
    supplement = exact_replace(supplement, 'the original shared-versus-independent horizon market experiment is still unexecuted.',
        'the matched four-head shared-versus-independent horizon market experiment is now complete, with its separate development decisions reported below.')
    mechanism_rows = []; candidate_rows = []; detail = []
    for h, entry in zip(HORIZONS, entries):
        assert entry['horizon'] == h and entry['independent_confirmation'] is False
        ci = [next(row for row in entry['intervals'] if row['primary'] == 'tied_last'
                   and row['reference'] == 'untied_last' and row['block'] == block) for block in (1, 5)]
        m = entry['mechanism']
        mechanism_rows.append([f'H{h}', pct(ci[0]['relative_improvement_pct']),
            sum(m['seed_positive'].values()), f"{m['positive_folds']}/5", interval_text(ci[0]), interval_text(ci[1]),
            'pass' if m['passed'] else 'fail'])
        checks = entry['candidate_checks']
        candidate_rows.append([f'H{h}', pct(entry['summary'][CANDIDATE]['improvement_pct_over_zero']),
            checks['three_positive_seeds_over_zero'], checks['four_positive_folds_over_zero'],
            checks['minimum_one_percent_over_zero'], checks['both_block_intervals_against_every_control'],
            'pass' if entry['candidate_performance_gate_passed'] else 'fail'])
        summary_rows = [[name.replace('_', ' '), value['origins'], f"{value['endpoint_mse']:.9g}",
            pct(value['improvement_pct_over_zero']), f"{value['positive_folds_over_zero']}/5"]
            for name, value in entry['summary'].items()]
        interval_rows = [[row['primary'].replace('_', ' '), row['reference'].replace('_', ' '), row['block'],
            pct(row.get('relative_improvement_pct')), interval_text(row), row['supports_improvement']]
            for row in entry['intervals']]
        detail.append(f"### S17.{HORIZONS.index(h)+1} Minute H{h} ({'primary' if h == 60 else 'supporting'})\n\n"+
            table(['Method', 'Origins', 'Endpoint MSE', 'Gain over zero', 'Positive folds'], summary_rows)+
            '\n\n'+table(['Primary', 'Reference', 'Block days', 'Gain', '95% interval', 'Supports gain'], interval_rows))
    methods_source = (BASE/'manuscript/SHARED_STUDY_METHODS_EN.md').read_text()
    methods_body = '## Matched shared and independent horizon functions'+methods_source.split(
        '## Matched shared and independent horizon functions', 1)[1].split('## Results to add after completion', 1)[0]
    methods_body = methods_body.replace('\n## ', '\n### ')
    if methods_body.startswith('## '): methods_body = '#'+methods_body
    appendix = f'''

## S15. Genuine four-head sharing experiment and numerical implementation

{methods_body}

Detached replay additionally verifies GPU-generated synthetic checkpoints on CPU at fixed rtol 2e-5 and atol 5e-7; its observed maximum absolute difference was below 7.4e-10. GPU replay requires exact equality on the specified environment. Hardware and kernel changes can prevent bitwise cross-device equality. These numerical checks remain engineering evidence.

## S16. Separate mechanism and sole-candidate decisions

The mechanism compares raw final tied versus raw final untied. Positive means lower tied error. Each row reports three seed directions, fold directions and two nominal paired block intervals. Only minute H60 is primary.

{table(['Horizon', 'Raw tied / untied gain', 'Positive seeds', 'Positive folds', 'Block1 interval', 'Block5 interval', 'Mechanism'], mechanism_rows)}

The sole candidate is tied-selected-shrink. Its four required checks are kept separate below. H60 must satisfy both interval lengths against all forty frozen controls; supporting horizons use all seven applicable controls. A threshold or control failure cannot be replaced by a favorable point estimate. All preceding failed methods remain retained.

{table(['Horizon', 'Gain over zero', '3 seeds', '4 folds', 'At least 1%', 'All controls / both blocks', 'Candidate'], candidate_rows)}

{mechanism_sentence} {candidate_sentence} {interpretation} No result in this minute-only study completes the original hourly-H8 objective or constitutes independent market confirmation. Existing exposed history remains development data, including all new supporting-horizon results. Nominal bootstrap intervals are not multiplicity-adjusted discovery claims.

## S17. Complete four-horizon endpoint and paired-interval tables

All saved summary rows, individual seeds, prediction-ensemble diagnostics and legacy H60 controls are included below. The candidate gate uses mean seed losses, not averaged forecasts. All 20 fold receipts additionally retain endpoint MSE, path MSE and endpoint MAE for their full registered method sets; those values remain in the hashed analysis rather than being redefined here. A positive interval describes reference-minus-primary error on exposed history. Undefined or degenerate intervals do not pass the gate.

{chr(10).join(detail)}

## S18. Completion, provenance and remaining research goal

The manuscript builder requires all thirty completed training receipts, the complete queue receipt, the 82-file analysis closure and a passed independent statistics review with matching source/input identities before reading these results. The independent reviewer reconstructs endpoint/path/absolute errors, fit-only mean control calibration and trading-day block statistics; it does not rerun a new market period. Model scalar provenance and the exact recorded shrink relation are checked; full inner-stop re-inference is not claimed by that statistics review.

The original title, author and school details, five temporal scales, twenty task table, annual sample limits, split-event sensitivity and completed negative results remain preserved. The author supplies the email before submission. Protected 2026Q3 remains sealed. A candidate that passes this development gate would still require a separately frozen eligible confirmation design. The overall mechanism-and-improvement research goal is not marked achieved by this manuscript or by minute-only development success. No external submission, public deposit, DOI or trading-profitability claim is made.
'''
    supplement += appendix
    for text in (main, supplement):
        assert 'shared-versus-independent horizon market experiment is still unexecuted' not in text
        assert 'shared-versus-independent-horizon market experiment remains unrun' not in text
    assert main.startswith('# '+TITLE+'\n')
    for value in ('Wang Yang', 'High School Affiliated to Shanghai Jiao Tong University',
                  'Shanghai 200439, China', '[Email to be completed by the author]'):
        assert value in main
    return main, supplement, {'primary_mechanism_gate_passed': mechanism,
        'primary_candidate_performance_gate_passed': candidate, 'primary_candidate_gain_pct': gain,
        'original_twenty_task_table_preserved': True, 'original_title_preserved': True,
        'Q3_values_read': False, 'independent_confirmation': False, 'research_goal_achieved': False}


def patch_render_layout(path):
    """Preserve formatter text while sizing the new compact table and its caption."""
    guard()
    assert path.resolve().is_relative_to((BASE/'manuscript/new_version').resolve()) or (
        path.parent.name.startswith('aapl_SYNTHETIC_manuscript_render_') and path.parent.parent == Path('/tmp'))
    from docx import Document
    from docx.shared import Mm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    formatter = load_module(V3B/'code/format_context_documents.py', 'shared_layout_formatter')
    before = formatter.inspect(path); document = Document(path); changed = False
    for paragraph in document.paragraphs:
        if paragraph.text.startswith('Table 3.'):
            paragraph.paragraph_format.keep_with_next = True
            paragraph.paragraph_format.keep_together = True
    for tab in document.tables:
        headings = [cell.text for cell in tab.rows[0].cells]
        if headings == ['H', 'Candidate gain', 'Folds', 'Mechanism', 'Candidate']:
            widths = [.08, .27, .13, .26, .26]; width = (178-5.08)/2
        elif headings == ['Primary', 'Reference', 'Block days', 'Gain', '95% interval', 'Supports gain']:
            widths = [.22, .31, .07, .11, .19, .10]; width = 178
        else: continue
        for index, column in enumerate(tab.columns): column.width = Mm(width*widths[index])
        for row in tab.rows:
            for index, cell in enumerate(row.cells): cell.width = Mm(width*widths[index])
        changed = True
    guard(); document.save(path); guard()
    assert formatter.inspect(path) == before
    synthetic_marker = path.parent.name.startswith('aapl_SYNTHETIC_manuscript_render_')
    if synthetic_marker:
        document = Document(path)
        header_parts = set(); footer_parts = set()
        for section in document.sections:
            headers = [section.header]; footers = [section.footer]
            if section.different_first_page_header_footer:
                headers.append(section.first_page_header); footers.append(section.first_page_footer)
            if document.settings.odd_and_even_pages_header_footer:
                headers.append(section.even_page_header); footers.append(section.even_page_footer)
            for header in headers:
                identity = str(header.part.partname)
                if identity in header_parts: continue
                header_parts.add(identity)
                paragraph = header.paragraphs[0]
                paragraph.text = 'SYNTHETIC RENDER ACCEPTANCE — GENERATED STATISTICS, NOT MARKET RESULTS'
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for footer in footers:
                identity = str(footer.part.partname)
                if identity in footer_parts: continue
                footer_parts.add(identity)
                footer.paragraphs[0].add_run('   SYNTHETIC — NOT MARKET RESULTS')
        guard(); document.save(path); guard()
        after = formatter.inspect(path)
        assert all(before[key] == after[key] for key in before if key != 'text')
        # inspect() reads document.xml, so headers and footers leave main prose
        # identical as well; retain this assertion to prevent content drift.
        assert before == after
    receipt_path = Path(str(path)+'.preservation.json'); receipt = read(receipt_path)
    receipt['shared_table_widths_and_caption_adjusted'] = changed
    receipt['synthetic_each_page_header_footer_marker'] = synthetic_marker
    write(receipt_path, receipt)


def render(folder, formatter_python):
    guard()
    for command in ('pandoc', 'libreoffice', 'pdfinfo', 'pdftotext', 'pdftoppm', 'pdffonts', 'fc-match'):
        assert shutil.which(command), f'Missing rendering command: {command}'
    assert formatter_python.is_file()
    audit = folder/'build_audit'; audit.mkdir()
    config = audit/'fontconfig.xml'
    text_write(config, '<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd"><fontconfig><include ignore_missing="yes">/etc/fonts/fonts.conf</include><dir>/mnt/c/Windows/Fonts</dir></fontconfig>')
    env = dict(os.environ, FONTCONFIG_FILE=str(config), OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='-1')
    font = command_output(['fc-match', 'Times New Roman'], env=env, text=True)
    assert 'times' in font.lower(), font
    documents = []
    for name, double in (('MAIN_PAPER', True), ('SUPPLEMENT', False)):
        guard()
        source = folder/f'{name}_EN.md'; raw = folder/f'{name}_RAW.docx'; output = folder/f'{name}_EN.docx'
        command_run(['pandoc', '--from=markdown+tex_math_single_backslash+autolink_bare_uris', '--to=docx',
            '--standalone', f'--resource-path={folder}', f'--output={raw}', str(source)], env=env)
        command = [str(formatter_python), str(V3B/'code/format_context_documents.py'), str(raw), str(output)]
        if not double: command.append('single')
        command_run(command, env=env)
        command_run([str(formatter_python), str(Path(__file__).resolve()), '--layout-docx', str(output)], env=env)
        profile = audit/f'lo_profile_{name.lower()}'
        command_run(['libreoffice', f'-env:UserInstallation={profile.as_uri()}', '--headless',
            '--convert-to', 'pdf', '--outdir', str(folder), str(output)], env=env, timeout=180)
        pdf = output.with_suffix('.pdf'); assert pdf.is_file()
        pdf_hash = sha(pdf)
        info = command_output(['pdfinfo', str(pdf)], text=True)
        pages = int(next(line.split(':', 1)[1] for line in info.splitlines() if line.startswith('Pages:')))
        command_run(['pdftotext', '-layout', str(pdf), str(folder/f'{name}_PDF_TEXT.txt')])
        # A new hash-named directory precludes stale previews from any earlier PDF.
        previews = folder/'page_previews'/name.lower()/pdf_hash[:16]; previews.mkdir(parents=True)
        command_run(['pdftoppm', '-scale-to', '1600', '-png', str(pdf), str(previews/'page')], env=env)
        assert sha(pdf) == pdf_hash
        images = sorted(previews.glob('page-*.png')); assert len(images) == pages
        marker_checks = []
        if folder.name.startswith('aapl_SYNTHETIC_manuscript_render_'):
            for page in range(1, pages+1):
                page_text = command_output(['pdftotext', '-f', str(page), '-l', str(page), '-layout', str(pdf), '-'], text=True)
                # One header and one footer, with no linked-section duplicates.
                assert page_text.count('SYNTHETIC RENDER ACCEPTANCE — GENERATED STATISTICS, NOT MARKET RESULTS') == 1, (name, page, 'header')
                assert page_text.count('SYNTHETIC — NOT MARKET RESULTS') == 1, (name, page, 'footer')
                marker_checks.append({'page': page, 'header_count': 1, 'footer_count': 1,
                    'individual_page_text_sha256': hashlib.sha256(page_text.encode()).hexdigest()})
        preview_manifest = {'pdf_sha256': pdf_hash, 'pdf': str(pdf),
            'images': {p.name: sha(p) for p in images}, 'individual_page_marker_checks': marker_checks}
        write(previews/'PREVIEW_SOURCE_MANIFEST.json', preview_manifest)
        fonts = command_output(['pdffonts', str(pdf)], text=True)
        text_write(folder/f'{name}_PDF_FONTS.txt', fonts)
        preservation = read(Path(str(output)+'.preservation.json'))
        assert preservation['passed'] is True
        documents.append({'name': name, 'pages': pages, 'source_sha256': sha(source), 'docx_sha256': sha(output),
            'pdf_sha256': pdf_hash, 'preview_pages': len(images), 'preview_directory': str(previews),
            'preview_manifest_sha256': sha(previews/'PREVIEW_SOURCE_MANIFEST.json'),
            'individual_page_marker_checks': marker_checks,
            'times_embedded': 'TimesNewRoman' in fonts, 'preservation': preservation})
        guard()
    return {'status': 'rendered_requires_visual_review', 'visual_review_complete': False, 'documents': documents,
        'formatter_sha256': sha(V3B/'code/format_context_documents.py'),
        'main_page_target_4_to_6': 4 <= documents[0]['pages'] <= 6}


def synthetic_stats():
    tasks = {}
    for h in HORIZONS:
        names = ('zero', 'fit_mean_raw', 'fit_mean_shrink')+METHODS
        summary = {name: {'origins': 100, 'endpoint_mse': 1. if name == 'zero' else 1.01,
            'improvement_pct_over_zero': 0. if name == 'zero' else -1.,
            'positive_folds_over_zero': 2} for name in names}
        intervals = [{'primary': 'tied_last', 'reference': 'untied_last', 'block': block,
            'status': 'descriptive', 'relative_improvement_pct': 0., 'relative_ci95_pct': [-1., 1.],
            'supports_improvement': False} for block in (1, 5)]
        tasks[f'minute_h{h}_future_log_return'] = {'horizon': h, 'summary': summary,
            'mechanism': {'passed': False, 'seed_positive': {str(s): False for s in SEEDS}, 'positive_folds': 2},
            'candidate_checks': {'three_positive_seeds_over_zero': False, 'four_positive_folds_over_zero': False,
                'minimum_one_percent_over_zero': False, 'both_block_intervals_against_every_control': False},
            'candidate_performance_gate_passed': False, 'intervals': intervals, 'independent_confirmation': False,
            'candidate_control_directions': {}, 'primary_for_study': h == 60}
    return {'candidate': CANDIDATE, 'primary_task': 'minute_h60_future_log_return', 'tasks': tasks,
        'Q3_values_read': False, 'independent_confirmation': False, 'research_goal_achieved': False,
        'primary_mechanism_gate_passed': False, 'primary_candidate_performance_gate_passed': False}


def self_test():
    global DEADLINE, STOP
    # No completion-gate bypass is exposed by the formal CLI.
    rows = []
    for mechanism, candidate in ((False, False), (True, False), (False, True), (True, True)):
        value = synthetic_stats(); primary = value['tasks']['minute_h60_future_log_return']
        primary['mechanism']['passed'] = value['primary_mechanism_gate_passed'] = mechanism
        primary['candidate_performance_gate_passed'] = value['primary_candidate_performance_gate_passed'] = candidate
        main, supplement, snapshot = compose(value)
        assert snapshot['primary_mechanism_gate_passed'] == mechanism
        assert snapshot['primary_candidate_performance_gate_passed'] == candidate
        assert main.count('### 4.6 Completed matched horizon-sharing study') == 1
        assert '## S18.' in supplement and 'Hourly H8 | -1.636% | -0.795%' in main
        rows.append({'mechanism': mechanism, 'candidate': candidate, 'text_branch_checked': True})
    with tempfile.TemporaryDirectory(prefix='aapl_manuscript_gate_') as temporary:
        root = Path(temporary)
        # A tempting results file must not be opened while completion/review are absent.
        (root/'analysis').mkdir(); (root/'analysis/STATS.json').write_text('MUST_NOT_BE_PARSED')
        try: completion_gate(root)
        except AssertionError: refused = True
        else: raise AssertionError('Incomplete results were accepted')
        fixture = load_module(BASE/'review/release_tool_acceptance.py', 'manuscript_fake_metadata_fixture')
        full = root/'full_fake_metadata'; fixture.market_fixture(full)
        stats = read(full/'analysis/STATS.json'); stats.update(synthetic_stats()); write(full/'analysis/STATS.json', stats)
        results = read(full/'analysis/RESULTS.json'); results['statistics_sha256'] = sha(full/'analysis/STATS.json')
        write(full/'analysis/RESULTS.json', results)
        analysis = read(full/'analysis/ANALYSIS_COMPLETE.json')
        analysis['statistics_sha256'] = sha(full/'analysis/STATS.json')
        analysis['results_sha256'] = sha(full/'analysis/RESULTS.json')
        analysis['files'] = {name: sha(full/name) for name in analysis['files']}
        write(full/'analysis/ANALYSIS_COMPLETE.json', analysis)
        write(full/'audit/COMPLETE.json', {'status': 'completed', 'bundles': 30, 'Q3_values_read': False})
        (full/'review').mkdir()
        shutil.copy2(BASE/'review/independent_statistics.py', full/'review/independent_statistics.py')
        review = {'status': 'passed', 'Q3_values_read': False, 'independent_confirmation': False,
            'research_goal_achieved': False, 'tasks': deepcopy(stats['tasks']),
            'fold_checks': [{'horizon': h, 'fold': f, 'status': 'passed'} for h in HORIZONS for f in range(1, 6)]}
        for field, name in (('rules_sha256', 'protocol/RULES.json'),
                            ('analysis_complete_sha256', 'analysis/ANALYSIS_COMPLETE.json'),
                            ('evaluation_freeze_sha256', 'protocol/EVALUATION_FREEZE.json'),
                            ('script_sha256', 'review/independent_statistics.py')):
            review[field] = sha(full/name)
        review_path = full/'review/INDEPENDENT_STATISTICS_REVIEW.json'; write(review_path, review)
        completion_gate(full)
        # In each failure the statistics JSON is deliberately unparsable. An
        # AssertionError demonstrates the gate stops before trying to parse it.
        (full/'analysis/STATS.json').write_text('MUST_NOT_BE_PARSED')
        gate_rejections = []
        for case in ('failed_review', 'review_identity_mismatch', 'missing_DONE'):
            altered = deepcopy(review); removed = None
            if case == 'failed_review': altered['status'] = 'incomplete'
            elif case == 'review_identity_mismatch': altered['script_sha256'] = '0'*64
            else:
                removed = next((full/'runs').glob('*/DONE.json'))
                contents = removed.read_text(); removed.unlink()
            write(review_path, altered)
            try: completion_gate(full)
            except AssertionError: gate_rejections.append(case)
            else: raise AssertionError(f'Incomplete review gate was accepted: {case}')
            if removed: removed.write_text(contents)
    with tempfile.TemporaryDirectory(prefix='aapl_manuscript_deadline_') as temporary:
        marker = Path(temporary)/'MUST_NOT_START'
        guard_checks = {}
        DEADLINE = 1.
        for name, operation in (
            ('external_command_after_deadline', lambda: command_run([sys.executable, '-c',
                'from pathlib import Path; import sys; Path(sys.argv[1]).touch()', str(marker)])),
            ('receipt_after_deadline', lambda: write(marker, {'status': 'passed'}))):
            try: operation()
            except InterruptedError: guard_checks[name] = 'rejected'
            else: raise AssertionError(name)
        DEADLINE = None; interrupted()
        try: write(marker, {'status': 'passed'})
        except InterruptedError: guard_checks['receipt_after_signal'] = 'rejected'
        else: raise AssertionError('Signal guard did not fire')
        STOP = False
        assert not marker.exists()
    result = {'status': 'passed', 'scope': 'synthetic statistics and immutable prior manuscript only',
        'market_outer_arrays_or_results_read': False, 'GPU': False, 'Q3_values_read': False,
        'four_mechanism_candidate_branches': rows, 'incomplete_gate_rejected_before_results': refused,
        'full_fake_30_bundle_review_gate_accepted': True, 'rejected_before_unparsable_results': gate_rejections,
        'deadline_and_signal_guard_checks': guard_checks,
        'self_tests_have_no_expired_market_deadline': DEADLINE is None,
        'previous_sources': SOURCE_IDENTITIES, 'script_sha256': sha(__file__)}
    write(BASE/'review/MANUSCRIPT_BUILDER_PREPARATION.json', result)
    print(json.dumps(result), flush=True)


def synthetic_render_acceptance(formatter_python):
    """Render marked generated statistics in /tmp, never the formal destination."""
    stats = synthetic_stats()
    for h in HORIZONS:
        task = stats['tasks'][f'minute_h{h}_future_log_return']
        for method in METHODS:
            for name in [f'{method}_seed_{s}' for s in SEEDS]+[f'{method}_prediction_ensemble']:
                task['summary'][name] = deepcopy(task['summary'][method])
        controls = ['zero', 'fit_mean_raw', 'fit_mean_shrink']+[m for m in METHODS if m.startswith('untied_')]
        if h == 60:
            for index in range(33):
                name = f'legacy_v3b_synthetic_archived_control_{index:02d}'
                controls.append(name); task['summary'][name] = deepcopy(task['summary']['zero'])
        for reference in controls:
            for block in (1, 5):
                task['intervals'].append({'primary': CANDIDATE, 'reference': reference, 'block': block,
                    'status': 'descriptive', 'relative_improvement_pct': -1.,
                    'relative_ci95_pct': [-2., 0.5], 'supports_improvement': False})
        assert len(task['summary']) == (76 if h == 60 else 43)
        assert len(task['intervals']) == (82 if h == 60 else 16)
    main_text, supplement, _ = compose(stats)
    folder = Path(tempfile.mkdtemp(prefix='aapl_SYNTHETIC_manuscript_render_'))
    banner = '**SYNTHETIC RENDER ACCEPTANCE ONLY: generated statistics; not market results or a submission draft.**\n\n'
    first, rest = main_text.split('\n\n', 1)
    (folder/'MAIN_PAPER_EN.md').write_text(first+'\n\n'+banner+rest)
    (folder/'SUPPLEMENT_EN.md').write_text(banner+supplement)
    shutil.copytree(V3B/'manuscript/figures', folder/'figures')
    report = render(folder, formatter_python)
    report.update(scope='explicitly marked synthetic statistics; temporary render only',
        temporary_directory=str(folder), formal_manuscript_generated=False,
        market_outer_arrays_or_results_read=False, GPU=False, Q3_values_read=False,
        generated_summary_counts=[len(stats['tasks'][f'minute_h{h}_future_log_return']['summary']) for h in HORIZONS],
        generated_interval_counts=[len(stats['tasks'][f'minute_h{h}_future_log_return']['intervals']) for h in HORIZONS],
        script_sha256=sha(__file__))
    write(BASE/'review/MANUSCRIPT_SYNTHETIC_RENDER.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--self-test-render', action='store_true', help='Render synthetic acceptance fixtures in /tmp only')
    parser.add_argument('--layout-docx', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--render', action='store_true', help='Also generate double-column DOCX/PDF and page previews')
    parser.add_argument('--formatter-python', type=Path, default=Path('/home/ayaka/.venvs/doc-tools/bin/python'))
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    assert set(os.sched_getaffinity(0)).issubset({24, 26})
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT): signal.signal(sig, interrupted)
    if args.layout_docx:
        if args.layout_docx.resolve().is_relative_to((BASE/'manuscript/new_version').resolve()):
            authorize_formal_deadline()
        patch_render_layout(args.layout_docx); return
    if args.self_test_render:
        synthetic_render_acceptance(args.formatter_python); return
    if args.self_test:
        self_test(); return
    authorize_formal_deadline()
    stats, review, provenance = completion_gate()
    main_text, supplement, snapshot = compose(stats, provenance['recorded_training_compute'])
    destination = BASE/'manuscript/new_version'
    assert not destination.exists(), 'Never overwrite a prior manuscript build'
    guard()
    destination.mkdir(parents=True)
    shutil.copytree(V3B/'manuscript/figures', destination/'figures')
    guard()
    shutil.copy2(V3B/'manuscript/AUTHOR_INFORMATION.json', destination/'AUTHOR_INFORMATION.json')
    for name in ('MAIN_PAPER_EN.md', 'SUPPLEMENT_EN.md'):
        guard()
        copy = destination/'preceding_version'/name; copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(V3B/'manuscript'/name, copy)
    text_write(destination/'MAIN_PAPER_EN.md', main_text)
    text_write(destination/'SUPPLEMENT_EN.md', supplement)
    snapshot.update(provenance=provenance, preceding_sources=SOURCE_IDENTITIES,
        reviewed_methods_sha256=sha(BASE/'manuscript/SHARED_STUDY_METHODS_EN.md'),
        manuscript_builder_sha256=sha(__file__), old_manuscripts_modified=False,
        independent_review_is_numerical_replication_only=True)
    write(destination/'RESULT_SOURCE_SNAPSHOT.json', snapshot)
    build = render(destination, args.formatter_python) if args.render else {
        'status': 'sources_written_not_rendered', 'visual_review_complete': False}
    write(destination/'DOCUMENT_BUILD.json', build)
    guard()
    print(json.dumps({'status': build['status'], 'destination': str(destination), 'snapshot': snapshot}), flush=True)


if __name__ == '__main__':
    main()
