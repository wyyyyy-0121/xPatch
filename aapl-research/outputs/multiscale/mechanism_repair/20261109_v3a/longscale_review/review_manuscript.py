"""Independent Markdown claim/table review; no manuscript mutation or training."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

B = Path(__file__).resolve().parents[1]
O = B.parent / '20261109_v3'
D = B / 'manuscript'
R = Path(__file__).resolve().parent

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def pick(rows, task, method):
    found = [r for r in rows if r['task'] == task and r['method'] == method]
    assert len(found) == 1, (task, method, len(found))
    return found[0]

def label(t):
    s, h, _, target = t.split('_', 3)
    return f'{s.capitalize()} H{h[1:]} ' + ('price' if target == 'close' else 'return')

def pc(x):
    return f'{x:+.3f}%'

def tables(text):
    out, current = [], []
    for line in text.splitlines() + ['']:
        if line.startswith('|'):
            current.append([v.strip() for v in line.strip('|').split('|')])
        elif current:
            assert all(set(v) <= {'-', ':'} for v in current[1])
            out.append({'header': current[0], 'rows': current[2:]})
            current = []
    return out

def main():
    mainp, suppp = D/'MAIN_PAPER_EN.md', D/'SUPPLEMENT_EN.md'
    text, supp = mainp.read_text(), suppp.read_text()
    bn_final='### 4.4 Fixed-weight normalization diagnostic' in text
    sources = [mainp, suppp, D/'SOURCE_AUDIT.json', O/'manuscript/MAIN_PAPER_EN.md',
        O/'analysis/CANDIDATE_RESULTS.json', O/'analysis/REPLICATION_RESULTS.json',
        O/'analysis/CAP_RESULTS.json', O/'analysis/STRUCTURAL_PROBES.json',
        O/'analysis/COMPARISON_AUDIT.json',O/'analysis/SCREENING_RESULTS.json',
        O/'protocol/JOBS.json',O/'protocol/REPLICATION_JOBS.json',O/'protocol/CANDIDATE_JOBS.json',
        B/'longscale_replication/analysis/RESULTS.json',
        B/'mean_controls/audit/RESULTS.json', B/'history_sensitivity/analysis/RESULTS.json']
    if bn_final:
        sources += [B/'bn_recalibration/RESULTS.json',B/'bn_recalibration/CONTROL_AUDIT.json',R/'BN_REVIEW.json']
    before = {str(p):sha(p) for p in sources}
    long = read(B/'longscale_replication/analysis/RESULTS.json')
    mean = read(B/'mean_controls/audit/RESULTS.json')
    hist = read(B/'history_sensitivity/analysis/RESULTS.json')
    old = read(O/'analysis/CANDIDATE_RESULTS.json')
    cap = read(O/'analysis/CAP_RESULTS.json')
    rep = read(O/'analysis/REPLICATION_RESULTS.json')
    probe = read(O/'analysis/STRUCTURAL_PROBES.json')
    issues, checks = [], {}
    def check(k, condition, detail=None):
        checks[k] = bool(condition)
        if not condition:
            issues.append({'severity':'error','check':k,'detail':detail})
    tasks = sorted(set(r['task'] for r in old['pooled']))
    lts = list(dict.fromkeys(r['task'] for r in long['pooled']))
    def candidate(t):
        return pick(long['pooled'],t,'repair_shrink') if t in lts else pick(old['pooled'],t,'candidate')
    mt, st = tables(text), tables(supp)
    expected_main = []
    for scale, horizons in [('annual',[12]),('quarterly',[3]),('monthly',[21]),('hourly',[1,4,8]),('minute',[5,15,30,60])]:
        for h in horizons:
            p,r = [candidate(f'{scale}_h{h}_{target}') for target in ('future_close','future_log_return')]
            expected_main.append([f'{scale.capitalize()} H{h}',pc(p['improvement_pct']),pc(r['improvement_pct']),f"{p['seed_count']}/{r['seed_count']}"])
    check('main_table1_all20tasks_gains_and_seed_counts',len(mt)==2 and mt[0]['rows']==expected_main)
    expected_cap=[]
    for model in ['lstm','xpatch']:
        vals=[next(r['endpoint_mse'] for r in cap['pooled'] if r['model']==model and r['method']==m) for m in ('default20','default60','price_baseline20')]
        expected_cap.append(['LSTM' if model=='lstm' else 'xPatch',f'{vals[0]:.2f}',f'{vals[1]:.2f}',f'{vals[2]:.4f}'])
    check('main_table2_cap_mse',mt[1]['rows']==expected_cap)
    expected=[]
    expected.append([[label(t)]+[pc(pick(long['pooled'],t,m)['improvement_pct']) for m in ('default_trained','default_shrink','repair_trained','repair_shrink')]+[f"{candidate(t)['positive_folds']}/5"] for t in lts])
    expected.append([[label(t)]+[pc(candidate(t)['seed_improvements_pct'][str(s)]) for s in (2026,2027,2028)]+[pc(pick(mean['pooled'],t,m)['improvement_pct']) for m in ('raw','shrink')] for t in lts])
    expected.append([[label(r['task']),str(r['block']),f"[{r['relative_ci95_pct'][0]:+.2f}, {r['relative_ci95_pct'][1]:+.2f}]%",str(r['supports_improvement'])] for r in long['paired_intervals'] if r['primary']=='repair_shrink' and r['reference']=='fit_mean_shrink'])
    expected.append([[label(t),str(candidate(t)['origins']),pc(pick(mean['pooled'],t,'raw')['improvement_pct']),pc(pick(mean['pooled'],t,'shrink')['improvement_pct']),pc(candidate(t)['improvement_pct'])] for t in tasks])
    expected.append([[label(t)]+[pc(pick(hist['pooled'],t,m)['improvement_over_baseline_pct']) for m in ('actual','fit_mean_history','flat_history')] for t in lts])
    expected.append([[label(r['task']),str(r['origins']),pc(r['methods']['repair_shrink']['improvement_pct']),pc(r['methods']['fit_mean_shrink']['improvement_pct'])] for r in long['sensitivity'] if r['rule']=='original_global_earliest_greedy'])
    phase=[]
    for t in lts:
        for rule in ('absolute_bar_phase','leave_origin_year_out'):
            rs=[r for r in long['sensitivity'] if r['task']==t and r['rule']==rule and r['origins']]
            phase.append([label(t),rule.replace('_',' '),str(len(rs)),f"{min(r['origins'] for r in rs)}–{max(r['origins'] for r in rs)}",f"{min(r['methods']['repair_shrink']['improvement_pct'] for r in rs):+.3f}% to {max(r['methods']['repair_shrink']['improvement_pct'] for r in rs):+.3f}%"])
    expected.append(phase)
    if bn_final:
        bn=read(B/'bn_recalibration/RESULTS.json');bc=read(B/'bn_recalibration/CONTROL_AUDIT.json')
        br=read(R/'BN_REVIEW.json')
        brow=[]
        for t,scope in [('minute_h60_future_log_return','Minute H60 return / 5 folds, 3 seeds'),('hourly_h8_future_log_return','Hourly H8 return / 2 folds, 1 seed only')]:
            brow.append([scope]+[pc(bn['tasks'][t]['improvement_over_zero_pct'][m]) for m in ('original','adapted','original_shrink','adapted_shrink')])
        expected.append(brow)
        audit=bc['tasks']['minute_h60_future_log_return']
        expected.append([[lab]+[str(audit['performance_modes'][mode]['checks'][k]) for k in ['minimum_one_percent_over_zero','all_three_seed_directions_positive','four_positive_fold_means','all_controls_positive_both_block_intervals']]+[str(audit['performance_modes'][mode]['performance_gate_passed'])] for lab,mode in [('Minute BN','adapted'),('Minute BN shrink','adapted_shrink')]])
        expected.append([[str(seed)]+[pc(audit['per_seed_improvement_over_zero_pct'][str(seed)][m]) for m in ('adapted','adapted_shrink')] for seed in (2026,2027,2028)])
        check('BN_independently_accepted',br['status']=='passed' and br['source_sha256']['bn_recalibration/RESULTS.json']==sha(B/'bn_recalibration/RESULTS.json') and br['source_sha256']['bn_recalibration/CONTROL_AUDIT.json']==sha(B/'bn_recalibration/CONTROL_AUDIT.json'))
        raw_reduction=100*(1-audit['pooled_mse']['adapted']/audit['pooled_mse']['repair_original_raw'])
        check('BN_raw_reduction_prose',f'{raw_reduction:.3f}%' in text and f'{raw_reduction:.6f}%' in supp)
        raws=[r for r in audit['paired_intervals'] if r['primary']=='adapted' and r['reference']=='repair_original_raw']
        shrinks=[r for r in audit['paired_intervals'] if r['primary']=='adapted_shrink' and r['reference']=='repair_original_shrink']
        check('BN_raw_local_intervals_support',len(raws)==2 and all(r['supports_improvement'] for r in raws))
        check('BN_shrink_intervals_cross_zero',len(shrinks)==2 and all(r['relative_ci95_pct'][0]<0<r['relative_ci95_pct'][1] for r in shrinks))
        check('BN_primary_failure_prose',all(audit['per_seed_improvement_over_zero_pct'][str(s)]['adapted_shrink']<0 for s in (2026,2027,2028)) and sum(r['adapted_shrink_gain_pct']>0 for r in audit['folds'])==1 and all(not row['performance_gate_passed'] for row in audit['performance_modes'].values()) and 'two-fold, single-seed diagnostic' in text)
    check('supplement_table_count',len(st)==len(expected))
    for i,e in enumerate(expected):
        check(f'supplement_table_{i+1}_all_rows',len(st)>i and st[i]['rows']==e)
    # Unconditional controls are deterministic and must match both independently saved sources.
    for t in lts:
        for a,b in [('fit_mean','raw'),('fit_mean_shrink','shrink')]:
            check(f'controls_consistency_{t}_{b}',abs(pick(long['pooled'],t,a)['improvement_pct']-pick(mean['pooled'],t,b)['improvement_pct'])<1e-10)
    title='Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales'
    block=f'# {title}\n\nWang Yang\n\nHigh School Affiliated to Shanghai Jiao Tong University\n\nShanghai 200439, China\n\n[Email to be completed by the author]\n'
    check('title_author_school_location_email_preserved',text.startswith(block))
    oldtext=(O/'manuscript/MAIN_PAPER_EN.md').read_text()
    check('main_references_verbatim',text.split('## References',1)[1]==oldtext.split('## References',1)[1])
    for name in ('MAIN_PAPER_EN.md','MAIN_PAPER_EN.pdf','MAIN_PAPER_EN.docx','SUPPLEMENT_EN.md','SUPPLEMENT_EN.pdf','SUPPLEMENT_EN.docx'):
        check(f'preceding_copy_{name}',sha(D/'preceding_version'/name)==sha(O/'manuscript'/name))
    build=read(O/'audit/DOCUMENT_BUILD.json')
    for doc in build['documents']:
        prefix=doc['document']+'_EN'
        for ext,key in [('md','source_sha256'),('pdf','pdf_sha256'),('docx','docx_sha256')]:
            check(f'original_v3_archive_document_{prefix}_{ext}',sha(O/'manuscript'/f'{prefix}.{ext}')==doc[key])
    metadata=read(D/'SOURCE_AUDIT.json')
    check('metadata_task_lists',sorted(metadata['all20_tasks'])==tasks and metadata['table1']==expected_main and metadata['supplemental_longscale_tasks']==lts)
    for rel,hashval in metadata['inputs'].items():
        p=Path(rel) if Path(rel).is_absolute() else B/rel
        check(f'metadata_hash_{rel}',p.exists() and sha(p)==hashval)
    check('original_primary_criteria_false',old['primary_tasks_both_pass'] is False)
    check('result_flags_no_success_or_independent_or_Q3',all(x.get('Q3_values_read') is False and x.get('independent_confirmation') is False for x in (long,mean,hist,old)))
    check('longscale_accounting',long['paired_cells']==180 and long['new_cells']==150 and len(long['rows'])==360 and len(long['pooled'])==66 and len(long['paired_intervals'])==120)
    check('mean_accounting',len(tasks)==20 and len(mean['fold_results'])==100)
    check('history_accounting',len(hist['rows'])==90 and len(hist['pooled'])==24 and len(hist['paired_intervals'])==24)
    check('cap_accounting',len(cap['comparisons'])==30 and sum(x['new_updates']>0 for x in cap['comparisons'])==17 and sum(x['new_updates']==0 for x in cap['comparisons'])==13)
    check('original_factorial_replication_count',rep['admitted_cells']==240 and len(rep['original_restorations'])==20)
    comparison=read(O/'analysis/COMPARISON_AUDIT.json')
    screen=read(O/'analysis/SCREENING_RESULTS.json')
    repjobs=read(O/'protocol/REPLICATION_JOBS.json')
    candjobs=read(O/'protocol/CANDIDATE_JOBS.json')
    screenids={r['id'] for r in read(O/'protocol/JOBS.json')['jobs']}
    repids={r['id'] for r in repjobs['jobs']}
    check('original_neural_fit_and_tiny_counts',all(comparison[k]==v for k,v in {'original_fits':400,'original_epoch_cap':81,'tiny_fits':390,'tiny_epoch_zero_selected':206,'tiny_training_ends_at_three_epochs':206}.items()))
    check('screen_count_and_default_restoration',screen['new_fits']==52 and len(screenids)==52 and len(screen['rows'])==156 and len(screen['original_restoration'])==12 and len(screen['tiny_prefix_checks'])==4)
    check('replication_reused_screen_count',len(repids)==240 and len(repids&screenids)==32)
    check('candidate_reuse_counts',len(candjobs['jobs'])==130 and candjobs['reused_cells']==45 and candjobs['new_cells']==85)
    for t,expected in [('minute_h60_future_log_return',9.60),('hourly_h8_future_log_return',44.11)]:
        con=next(c for c in rep['contrasts'] if c['task']==t and c['primary']=='anchor_baseline' and c['reference']=='anchor_random' and c['mode']=='trained')
        check(f'initialization_reduction_{t}',round(con['pooled_improvement_over_reference_pct'],2)==expected and con['positive_folds']==5 and all(con['checks'].values()))
    baseline=[r for r in probe['rows'] if r['variant'].endswith('baseline')]
    check('first_update_zero_backbone_gradient_120of120',len(baseline)==120 and all(r['first_updates'][0]['backbone_all_zero'] for r in baseline))
    anual=[r for r in long['paired_intervals'] if r['task'].startswith('annual') and r['primary']=='repair_shrink' and r['reference']=='fit_mean_shrink']
    check('annual_vs_mean_intervals_cross_zero',len(anual)==4 and all(r['relative_ci95_pct'][0]<0<r['relative_ci95_pct'][1] for r in anual))
    flat=[r for r in hist['paired_intervals'] if r['task'].startswith('annual') and r['primary']=='actual' and r['reference']=='flat_history']
    check('annual_actual_vs_flat_intervals_favor_flat',len(flat)==4 and all(r['relative_ci95_pct'][1]<0 and r['difference_ci95'][1]<0 for r in flat))
    check('research_limits_explicit',all(v in text+supp for v in ['2026Q3 remains unscored','four paths','out of distribution','independent confirmation remain unfinished','shared-versus-independent']))
    rawhour=pick(mean['pooled'],'hourly_h8_future_log_return','raw')['improvement_pct']
    if 'The minute-H60 and hourly-H8 mean controls also fail to beat zero.' in supp and rawhour>0:
        issues.append({'severity':'wording_error','check':'S4_all_mean_controls_fail_zero','detail':f'Raw hourly-H8 mean return gain is {rawhour:+.9f}%; only both shrunk controls fail to beat zero. Neither raw nor shrunk meets original 1% criterion. Narrow the sentence before release.'})
    check('review_sources_unchanged_during_run',before=={str(p):sha(p) for p in sources})
    bn_pending='still in progress at this draft' in supp or 'final narrative and gates require' in supp
    errors=[i for i in issues if i['severity']=='error']
    renderfile=R/'RENDERED_DOCUMENTS_REVIEW.json'
    render_ok=False
    if renderfile.exists():
        rendered=read(renderfile)
        render_ok=rendered['status']=='passed' and all(d['source_sha256']==sha(D/(d['name']+'_EN.md')) for d in rendered['documents'])
    status=('numerics_issues' if issues else 'numerics_and_documents_passed' if render_ok else 'numerics_passed_render_pending') if bn_final else ('draftnumerics_issues_BN_pending' if issues else 'draftnumerics_passed_BN_pending')
    resolved=[]
    if not any(i['check']=='S4_all_mean_controls_fail_zero' for i in issues):
        resolved.append({'check':'S4_all_mean_controls_fail_zero','status':'resolved_by_parent_before_final_review','original_issue':'The original draft said both primary-task mean controls fail to beat zero; the raw hourly-H8 control is slightly positive.','verified_raw_hourly_H8_gain_pct':rawhour})
    receipt={'status':status,'checks':checks,'issues':issues,'resolved_issues':resolved,'tables_verified':{'main_tables':len(mt),'main_task_results':20,'supplement_tables':len(st),'supplement_data_rows':sum(len(t['rows']) for t in st)},
      'history_flat_intervals_verified':flat,'BN_final_result_in_scope':bn_final,'BN_pending_in_draft':bn_pending,
      'rendered_document_acceptance':'RENDERED_DOCUMENTS_REVIEW.json:passed' if render_ok else 'pending_parent_build','original_v3_unchanged':all(v for k,v in checks.items() if k.startswith('original_v3_archive_document')),
      'research_goal_achieved':False,'independent_confirmation':False,'Q3_values_read':False,
      'source_sha256':before,'reviewer_script_sha256':sha(Path(__file__)), 'at_utc':datetime.now(timezone.utc).isoformat()}
    (R/'MANUSCRIPT_REVIEW.json').write_text(json.dumps(receipt,indent=2)+'\n')
    scope='BN 最终数字与全部控制门槛已核验，DOCX 来源保真和 PDF 表格数字验收已通过；逐页视觉验收由主代理另行记录。' if bn_final and render_ok else 'BN 最终数字已核验，渲染验收待完成。' if bn_final else 'BN 最终结论和渲染文档验收待完成。'
    lines=['# v3a 稿件独立数字审阅','',f'状态：{status}。审阅覆盖 Markdown 来源、数字、表格与限制性表述；{scope}','']
    lines += [f"- {i['check']}: {i['detail']}" for i in issues] or ['数字与表格未发现问题。']
    (R/'MANUSCRIPT_ISSUES_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':status,'checks':len(checks),'failed_checks':len(errors),'issues':issues,'tables_verified':receipt['tables_verified']},indent=2))

if __name__=='__main__':
    main()
