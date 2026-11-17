"""Combine the existing manuscript and verified exploratory results in a private draft."""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path

from multiscale.bars import file_digest
from multiscale.spec import OUT


def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('Expected exactly one manuscript fragment: '+old[:90])
    return text.replace(old,new,1)


def main():
    draft=OUT/'PAPER_DRAFT_EN.md'
    appendices=sorted((OUT/'private_manuscript_appendices').glob('*_PRIVATE_PRELIMINARY_RESULTS_EN.md'))
    coverages=sorted((OUT/'private_coverage').glob('*.csv'))
    if not appendices or not coverages:raise ValueError('A private appendix and coverage snapshot are required')
    appendix=appendices[-1];coverage=coverages[-1]
    body=draft.read_text(encoding='utf8');results=appendix.read_text(encoding='utf8')
    with coverage.open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    completed=sum(int(r['completed_full_folds'])==5 for r in rows)
    if len(rows)!=20 or completed<12:raise ValueError('Coverage is inconsistent with the development snapshot')
    original_notice=('**Working manuscript: not ready for submission.** The long-history data source and its publication rights have not been admitted, '
                     'the experimental protocol is not frozen, and no formal multi-scale forecasts have been scored. '
                     'Bracketed items require evidence from completed work; they are not results.')
    new_notice=('**PRIVATE WORKING MANUSCRIPT — NOT FOR SUBMISSION OR DISTRIBUTION.** '
                'This version combines previously exposed, local exploratory experiments with the working methods draft. '
                'Account-specific data and publication rights, a final protocol, and independent confirmation remain unresolved. '
                'Bracketed items identify work that has not been completed.')
    body=replace_once(body,original_notice,new_notice)
    abstract_start=body.index('## Abstract\n');abstract_end=body.index('\n## 1. Introduction',abstract_start)
    abstract=('## Abstract\n\n'
              'Forecast accuracy for one stock can vary across horizons and between price-level and return targets. '
              'This private development study compares persistence, drift, seasonal repetition, ridge regression, an MLP, GRU, LSTM, '
              'and an adapted xPatch model on historical Apple Inc. (AAPL) observations across annual, quarterly, monthly, hourly, '
              'and minute tasks. At this snapshot, hourly, monthly, and quarterly tasks and a user-approved short-history annual '
              'variant have five chronological development folds; minute tasks remain incomplete. Simple persistence controls '
              'are competitive in several observed tasks. An exploratory split-neutral monthly return comparison also varies '
              'substantially across three random seeds. The annual sample is small and forecast paths overlap. '
              'These already-exposed results are descriptive, not independent confirmation, and source-specific publication '
              'rights remain unverified. The full matrix, final protocol, and any independent evaluation are outstanding.\n')
    body=body[:abstract_start]+abstract+body[abstract_end:]
    body=replace_once(body,'All substantive findings are reserved for the experiments. **[Insert a concise statement of contributions once the data and completed comparisons are known.]**',
                      'This working snapshot reports the completed development comparisons without claiming independent confirmation. '
                      '**[State the final contribution only after the remaining work and literature review are complete.]**')
    body=replace_once(body,'A separate Alpaca daily candidate covers January 2016 through June 2026, which is shorter than the original fifteen-year minimum for the annual task.',
                      'A separate Alpaca daily candidate covers January 2016 through June 2026, shorter than the original fifteen-year annual minimum. '
                      'For local exploratory work, the user accepted an approximately ten-year history and a twelve-month annual lookback; this is a deviation from the original design.')
    body=replace_once(body,'The current executor uses \\(L=60,36,252,96,480\\) bars for the annual, quarterly, monthly, hourly, and minute profiles, respectively; these choices remain subject to the final protocol freeze.',
                      'The current exploratory runs use \\(L=12,36,252,96,480\\) bars for the short-history annual, quarterly, monthly, hourly, and minute profiles, respectively; '
                      'the annual choice differs from the original 60-month lookback, and all choices remain subject to a final protocol.')
    body=replace_once(body,'The draft training budget and regularization grid are provisional until the protocol is frozen. **[Insert final architecture settings, optimizer, budgets, realized updates, stopping criteria, and seeds.]**',
                      'The current independent development runs use Adam with learning rate 0.001, batch size 128, at most 20 epochs, '
                      'patience 3, and ridge penalties 0.1, 1, and 10. The primary runs use seed 2026; the post-hoc split-neutral monthly '
                      'comparison additionally uses seeds 2027 and 2028. Realized update counts and final training settings remain to be documented.')
    body=replace_once(body,'A candidate quarterly warm-up split has been drafted to increase the first fitting sample, but it is not frozen and its small size would still limit inference.',
                      'For the local quarterly and short-history annual runs, a count-based warm-up required at least 24 inner-fit, '
                      '4 inner-stop, and 6 validation groups per fold. These are exploratory designs and their small samples limit inference.')
    body=replace_once(body,'These are draft settings to be frozen with the data and evaluation protocol before any outer scores; they do not establish stationarity or independent observations.',
                      'The reported development intervals are descriptive and were examined after development scores were available. '
                      'Any future confirmation requires a separately frozen protocol; block lengths do not establish stationarity or independent observations.')
    start=body.index('## 5. Experimental results\n');end=body.index('\n## 6. Limitations and conclusion',start)
    result_start=results.index('## Cumulative log-return target\n')
    result_end=results.index('\n## Outstanding work',result_start)
    sections=results[result_start:result_end]
    sections='\n'.join(('### '+line[3:]) if line.startswith('## ') else line for line in sections.splitlines())
    replacement=('## 5. Experimental results\n\n'
                 f'**Private development snapshot:** {completed} of 20 tasks have five completed development folds, including two '
                 'short-history annual tasks. The minute matrix and independent confirmation are incomplete. '
                 'The optional shared-versus-separate multi-horizon comparison has not been run. '
                 'All reported percentages compare endpoint MSE with the matched persistence or zero-return baseline; they are not investment returns.\n\n'
                 +sections+'\n')
    body=body[:start]+replacement+body[end:]
    body=replace_once(body,'**[Write the conclusion from completed, independently checked evidence; explicitly report negative results and unfinished comparisons.]**',
                      'In this exposed development snapshot, simple baselines are competitive and some model rankings vary substantially '
                      'between time folds and random seeds. A split-neutral target reduces the mechanical effect of the 2020 split on the '
                      'monthly baseline but does not establish a stable model advantage. These observations cannot be generalized as '
                      'independent evidence. **[Complete the conclusion after the remaining comparisons and any independent evaluation.]**')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    parent=OUT/'private_working_manuscripts';parent.mkdir(parents=True,exist_ok=True)
    directory=parent/stamp;directory.mkdir(exist_ok=False)
    path=directory/'PRIVATE_WORKING_MANUSCRIPT_EN.md'
    with path.open('x',encoding='utf8') as f:f.write(body)
    audit={'status':'private_working_manuscript_created_not_for_submission','created_at_utc':datetime.now(timezone.utc).isoformat(),
           'source_draft':str(draft),'source_draft_sha256':file_digest(draft),
           'source_appendix':str(appendix),'source_appendix_sha256':file_digest(appendix),
           'source_coverage':str(coverage),'source_coverage_sha256':file_digest(coverage),
           'five_fold_development_tasks_including_short_annual':completed,'markdown_sha256':file_digest(path),
           'script_sha256':file_digest(Path(__file__)),'independent_test':False,'publication_rights':'not confirmed',
           'limitations':['Results are a dated development snapshot; running tasks are not included.',
                          'Source publication rights, final protocol, independent confirmation, author details and venue remain unresolved.']}
    with (directory/'PRIVATE_MANUSCRIPT_AUDIT.json').open('x',encoding='utf8') as f:
        json.dump(audit,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps({'status':audit['status'],'path':str(path),'directory':str(directory),
                      'sha256':audit['markdown_sha256'],'five_fold_tasks':completed},ensure_ascii=False))


if __name__=='__main__':main()
