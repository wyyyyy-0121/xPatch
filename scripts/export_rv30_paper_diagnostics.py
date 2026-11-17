"""Export saved historical diagnostics for the paper; no training or new tests."""
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'research/paper'
OUT = PAPER / 'diagnostics'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    evidence = json.loads((PAPER / 'evidence.json').read_text())
    OUT.mkdir(exist_ok=True)
    annual, seeds, clocks, calibration = [], [], [], []
    sources = {}
    for phase in ('development', 'confirmation'):
        entry = evidence['inputs'][phase]
        path = ROOT / entry['path']
        if digest(path) != entry['sha256']:
            raise ValueError('Saved evaluation differs from paper evidence: ' + phase)
        sources[phase] = entry
        evaluation = json.loads(path.read_text())
        if evaluation['errors'] or not evaluation['historical_exposure']:
            raise ValueError('Unresolved or incorrectly described historical evaluation')
        for key, description in evaluation['descriptions'].items():
            panel, model = key.split('/')
            baseline = evaluation['descriptions'][panel + '/har']
            comparison = evaluation['comparisons'].get(key, {})
            for segment, values in description['year_segment'].items():
                base = baseline['year_segment'][segment]['qlike']
                annual.append(dict(phase=phase, information_set=panel, model=model,
                    segment=segment, **values,
                    qlike_gain_vs_same_panel_HAR_pct=100 * (1 - values['qlike'] / base)))
            for clock, values in description['clock'].items():
                base = baseline['clock'][clock]['qlike']
                clocks.append(dict(phase=phase, information_set=panel, model=model,
                    clock=clock, **values,
                    qlike_gain_vs_same_panel_HAR_pct=100 * (1 - values['qlike'] / base)))
            for seed, values in comparison.get('seed_differences', {}).items():
                for segment, difference in values.items():
                    seeds.append(dict(phase=phase, information_set=panel, model=model,
                        seed=seed, segment=segment, mean_qlike_difference_vs_HAR=difference))
            mz = description['mincer_zarnowitz']
            if mz['identified']:
                covariance = mz['hac_covariance']
                calibration.append(dict(phase=phase, information_set=panel, model=model,
                    intercept=mz['intercept'], slope=mz['slope'],
                    intercept_hac_se=math.sqrt(max(covariance[0][0], 0)),
                    slope_hac_se=math.sqrt(max(covariance[1][1], 0)),
                    covariance_intercept_slope=covariance[0][1],
                    hac_lags=mz['hac_lags'], sessions=mz['sessions']))
    tables = {}
    for name, rows in [('annual.csv', annual), ('seeds.csv', seeds),
                       ('clock.csv', clocks), ('calibration.csv', calibration)]:
        path = OUT / name
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, rows[0], lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        tables[str(path.relative_to(ROOT))] = {'sha256': digest(path), 'rows': len(rows)}
    result = dict(scope='export_of_saved_historical_diagnostics_not_new_inference',
        source=evidence['research_signature'], inputs=sources, tables=tables,
        derived_values=['relative QLIKE changes', 'square roots of saved HAC variances'],
        new_hypothesis_tests=False, new_training=False,
        observer_scores_read=False,
        exporter={'path': str(Path(__file__).relative_to(ROOT)), 'sha256': digest(Path(__file__))})
    (OUT / 'evidence.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'tables': tables, 'new_hypothesis_tests': False}))


if __name__ == '__main__':
    main()
