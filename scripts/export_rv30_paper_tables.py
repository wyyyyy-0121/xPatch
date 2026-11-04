"""Export manuscript tables from saved results without training or rescoring.

Run from any directory with the project's existing Python environment. These
exports retain the original model family, historical-exposure disclosure and
failed gates. They are working paper material, not final model acceptance.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'research/paper'
SIGNATURE = '65a4daf1ebacb8a7dedca5e866010bd94cda25acc7124509ec81d1c14dc2fbf8'
RUN = Path('artifacts/rv30') / SIGNATURE / 'development'
MODELS = ('last_rv', 'har', 'ridge', 'lightgbm', 'xpatch', 'chronos2')
INPUTS = {
    'development': RUN / 'evaluation.json',
    'confirmation': RUN / 'confirmation/evaluation.json',
    'development_round_1': RUN / 'verify_1_replay_correction/summary.json',
    'development_round_2': RUN / 'verify_2/summary.json',
    'confirmation_round_1': RUN / 'confirmation/verify_1/summary.json',
    'confirmation_round_2': RUN / 'confirmation/verify_2/summary.json',
    'search_coverage': RUN / 'search_coverage.json',
    'data': Path('delivery/data.json'),
    'protocol': Path('RESEARCH_PROTOCOL.md'),
    'historical_returns': Path('finance_v2/results_summary.json'),
    'execution_performance': Path('artifacts/rv30/setup/delayed_sip_20261104/throughput.json'),
}


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    with (ROOT / path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def csv_table(name, rows):
    if not rows:
        raise ValueError('An empty manuscript table must not replace a real result')
    with (OUTPUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def extract(evaluation, phase):
    if evaluation['signature'] != SIGNATURE or evaluation['errors']:
        raise ValueError('Wrong signature or unresolved evaluation errors')
    if not evaluation['historical_exposure']:
        raise ValueError('Historical exposure must be disclosed')
    rows, quarters = [], []
    for panel in ('aapl', 'market'):
        baseline = evaluation['descriptions'][panel + '/har']['qlike']
        for name in MODELS:
            key = panel + '/' + name
            if key not in evaluation['descriptions']:
                continue  # Historical confirmation only admits prior gate passers.
            d = evaluation['descriptions'][key]
            c = evaluation['comparisons'].get(key)
            if c is not None:
                if not math.isclose(d['qlike'] - baseline, c['mean_difference'], abs_tol=1e-12):
                    raise ValueError('Reported loss and paired comparison disagree: ' + key)
                expected_pass = all(c['gate_checks'].values())
                if c['passed'] is not expected_pass:
                    raise ValueError('A gate result differs from its component checks')
            blocks = {b['block_sessions']:b for b in c['bootstraps']} if c else {}
            rows.append({
                'phase':phase, 'information_set':panel, 'model':name,
                'qlike':d['qlike'], 'log_rv_mse':d['log_rv_mse'],
                'qlike_improvement_vs_same_panel_HAR_pct':100 * (1 - d['qlike'] / baseline),
                'qlike_difference_vs_HAR':c['mean_difference'] if c else '',
                'holm_p':c['holm_p'] if c else '',
                'ci95_day_lower':blocks[1]['ci95'][0] if c else '',
                'ci95_day_upper':blocks[1]['ci95'][1] if c else '',
                'ci95_five_day_lower':blocks[5]['ci95'][0] if c else '',
                'ci95_five_day_upper':blocks[5]['ci95'][1] if c else '',
                'improving_quarters':sum(v < 0 for v in c['quarter_differences'].values()) if c else '',
                'total_quarters':len(c['quarter_differences']) if c else '',
                'passed_all_registered_gates':c['passed'] if c else 'reference',
                'failed_gate_checks':';'.join(k for k,v in c['gate_checks'].items() if not v) if c else '',
                'coverage_80':d['coverage_80'], 'coverage_90':d['coverage_90'],
                'sessions':d['sessions'], 'origins':d['origins'],
            })
            for quarter, q in d['quarter'].items():
                quarters.append({'phase':phase, 'information_set':panel, 'model':name,
                    'quarter':quarter, 'qlike':q['qlike'], 'log_rv_mse':q['log_rv_mse'],
                    'sessions':q['sessions'], 'origins':q['origins']})
    return rows, quarters


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    values = {key:read(path) for key,path in INPUTS.items() if path.suffix == '.json'}
    for key in ('development_round_1','development_round_2','confirmation_round_1','confirmation_round_2'):
        if values[key]['passed'] is not True:
            raise ValueError('An existing independent verification did not pass: ' + key)
    coverage = values['search_coverage']
    if not coverage['passed'] or coverage['errors']:
        raise ValueError('The search coverage check has unresolved errors')
    development, dq = extract(values['development'], '2023Q1_2025Q2')
    confirmation, cq = extract(values['confirmation'], '2025H2_2026H1')
    if len(development) != 12 or len(confirmation) != 4:
        raise ValueError('Do not silently export a partial model family')
    csv_table('development.csv', development)
    csv_table('confirmation.csv', confirmation)
    csv_table('quarterly.csv', dq + cq)
    status = read(Path('delivery/status.json'))
    evidence = {'created_utc':datetime.now(timezone.utc).isoformat(), 'research_signature':SIGNATURE,
        'scope':'extraction_of_saved_verified_results_not_new_statistical_validation',
        'historical_exposure':True, 'independent_blind_holdout':False,
        'public_preregistration_established':False, 'original_protocol_signed_locally':True,
        'timely_forward_results_available':False, 'trading_profitability_established':False,
        'data_start':values['data']['start'], 'data_end':values['data']['end'],
        'development_sessions':development[0]['sessions'], 'development_origins_per_model':development[0]['origins'],
        'confirmation_sessions':confirmation[0]['sessions'], 'confirmation_origins_per_model':confirmation[0]['origins'],
        'completed_search_groups_excluding_fixed_last_rv':len(coverage['groups']),
        'inner_jobs':sum(g['total_inner_jobs'] for g in coverage['groups']),
        'runtime_snapshot':{key:status.get(key) for key in ('stage','updated_utc',
            'delayed_replay_complete_sessions','prospective_complete_sessions','last_error')},
        'inputs':{key:{'path':str(path),'sha256':digest(path)} for key,path in INPUTS.items()},
        'exporter':{'path':str(Path(__file__).relative_to(ROOT)),
            'sha256':digest(Path(__file__).relative_to(ROOT))},
        'tables':{name:digest(Path('research/paper')/name) for name in
            ('development.csv','confirmation.csv','quarterly.csv')}}
    (OUTPUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    lines = ['# 已完成实验的论文结果表', '',
        '以下数字从已保存的评价结果直接导出；没有重训、追加检验或查看前瞻累计分数。', '',
        '主指标 QLIKE 越小越好；相对改善以同信息集 HAR 为分母，不能解释成收益率或方向准确率。', '',
        '## 开发期：2023Q1—2025Q2', '',
        '共 602 个合格会话，每模型 207,235 个起点；采用交易日等权平均，分钟样本相互重叠。', '',
        '| 信息集 | 模型 | QLIKE | 相对 HAR 改善 | Holm p | 改善季度 | 全部门槛 |',
        '|---|---|---:|---:|---:|---:|---|']
    for row in development:
        reference = row['model']=='har'
        lines.append('| {information_set} | {model} | {qlike:.6f} | {improvement} | {p} | {quarters} | {gate} |'.format(
            **row, improvement='—' if reference else f"{row['qlike_improvement_vs_same_panel_HAR_pct']:.2f}%",
            p='—' if reference else f"{row['holm_p']:.6g}",
            quarters='—' if reference else f"{row['improving_quarters']}/{row['total_quarters']}",
            gate='参考基线' if reference else ('通过' if row['passed_all_registered_gates'] else '未通过')))
    lines += ['', '## 历史确认期：2025H2—2026H1', '',
        '共 251 个合格会话，每模型 86,306 个起点；只确认开发门槛通过者及其相应 HAR。', '',
        '| 信息集 | HAR QLIKE | Ridge QLIKE | 相对改善 | Holm p | 五会话块 95% CI（Ridge−HAR） |',
        '|---|---:|---:|---:|---:|---|']
    for panel in ('aapl','market'):
        rows = {row['model']:row for row in confirmation if row['information_set']==panel}
        ridge, har = rows['ridge'], rows['har']
        lines.append(f"| {panel} | {har['qlike']:.6f} | {ridge['qlike']:.6f} | "
            f"{ridge['qlike_improvement_vs_same_panel_HAR_pct']:.2f}% | {ridge['holm_p']:.6g} | "
            f"[{ridge['ci95_five_day_lower']:.6f}, {ridge['ci95_five_day_upper']:.6f}] |")
    lines += ['', '## 解释边界', '',
        '- aapl 为仅使用 AAPL 历史信息；market 加入 SPY/QQQ。两者都预测 AAPL，不能称为三个股票目标的验证。',
        '- AAPL LightGBM 的总体改善显著，但年度及种子稳定性未过，不能将“显著”写成通过全部预设门槛。',
        '- xPatch 在 AAPL 面板平均 QLIKE 改善约 2.97%，Holm p 约 0.330；未通过全部门槛。没有证据支持写成新架构取得领先。',
        '- 两个面板各自相对其 HAR 的检验，不能替代 market/Ridge 对 aapl/Ridge 的直接显著性检验。',
        '- 历史期间已有项目曝光；按时间切分的样本外评价仍是回顾性研究，不能称为未触碰盲测或外部验证。',
        '- 本地双轮回放验证计算与记录，不等于独立研究者、独立供应商或外部实验室复现。',
        '- 原 12 分钟收益任务的阴性结论独立保留，RV30 是不同目标，不可混写或称为同一目标性能翻转。',
        '- 15 分钟延迟运行用于新的影子观察，实际生成时间不倒填；目前没有完整观察日成绩或交易盈利证据。', '',
        '精确小数、置信区间、未通过的门槛及来源指纹见同目录 CSV 和 evidence.json。', '']
    (OUTPUT/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({'passed':True,'development_rows':len(development), 'confirmation_rows':len(confirmation),
        'quarter_rows':len(dq+cq),'directory':str(OUTPUT),'no_training_or_scoring_performed':True}))


if __name__ == '__main__':
    main()
