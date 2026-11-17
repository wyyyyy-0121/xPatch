"""Plot the saved five-fold endpoint-error ratios without selecting winners."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


MODELS = ('drift', 'seasonal', 'ridge', 'mlp', 'gru', 'lstm', 'xpatch')
LABELS = ('Drift', 'Seasonal', 'Ridge', 'MLP', 'GRU', 'LSTM', 'xPatch')
TASKS = (('annual', 12, 'Annual H12'), ('quarterly', 3, 'Quarterly H3'),
         ('monthly', 21, 'Monthly H21'), ('hourly', 1, 'Hourly H1'),
         ('hourly', 4, 'Hourly H4'), ('hourly', 8, 'Hourly H8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.metrics.open(encoding='utf8', newline='') as f:
        rows = list(csv.DictReader(f))
    lookup = {(r['profile'], int(r['horizon']), r['target'], r['model']): r for r in rows}
    if len(rows) != len(lookup) or len(rows) != 96 or any(r['folds'] != '5' for r in rows):
        raise ValueError('Expected all eight models for twelve complete five-fold tasks')
    panels, cells = [], []
    for target in ('future_close', 'future_log_return'):
        matrix = []
        for profile, horizon, _ in TASKS:
            baseline = float(lookup[profile, horizon, target, 'last_close']['endpoint_mse'])
            if not math.isfinite(baseline) or baseline <= 0:
                raise ValueError('Baseline error is undefined')
            values = []
            for model in MODELS:
                record = lookup[profile, horizon, target, model]
                value = float(record['endpoint_mse']) / baseline
                if not math.isfinite(value) or value <= 0:
                    raise ValueError('Model error ratio is undefined')
                if not math.isclose(1 - value, float(record['relative_improvement']), abs_tol=1e-12):
                    raise ValueError('Error ratio differs from the saved relative improvement')
                values.append(value)
                cells.append({'profile': profile, 'horizon': horizon, 'target': target,
                              'model': model, 'endpoint_mse_ratio': value, 'origins': int(record['origins'])})
            matrix.append(values)
        panels.append(np.asarray(matrix))
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.35), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=math.log10(0.5), vcenter=0, vmax=3.5)
    for ax, ratios, title in zip(axes, panels, ('Nominal close price', 'Cumulative log return'), strict=True):
        image = ax.imshow(np.log10(ratios), norm=norm, cmap='RdBu_r', aspect='auto')
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xticks(range(len(MODELS)), LABELS, rotation=30, ha='right')
        ax.set_yticks(range(len(TASKS)), [t[2] for t in TASKS])
        ax.tick_params(axis='both', length=0)
        ax.set_xticks(np.arange(-0.5, len(MODELS)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(TASKS)), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.8)
        ax.tick_params(which='minor', bottom=False, left=False)
        for (y, x), value in np.ndenumerate(ratios):
            level = norm(math.log10(value))
            label = (f'{value:.3f}' if 0.9 < value < 1.1 else
                     f'{value:.2f}' if value < 100 else f'{value:.0f}')
            ax.text(x, y, label, ha='center', va='center', fontsize=8,
                    color='white' if level < 0.18 or level > 0.82 else '#181818')
        for spine in ax.spines.values():
            spine.set_visible(False)
    ticks = [0.5, 1, 2, 10, 100, 1000]
    colorbar = fig.colorbar(image, ax=axes, fraction=0.032, pad=0.025,
                           ticks=np.log10(ticks), shrink=0.9)
    colorbar.ax.set_yticklabels([str(x) for x in ticks])
    colorbar.set_label('Endpoint MSE / persistence (log color scale)', labelpad=10)
    fig.suptitle('Historical development: twelve complete five-fold tasks', fontsize=13)
    paths = []
    for extension in ('png', 'pdf', 'svg'):
        path = args.output / ('completed_task_error_ratios.' + extension)
        fig.savefig(path, dpi=220, facecolor='white')
        paths.append(path)
    plt.close(fig)
    values = args.output / 'completed_task_error_ratios.csv'
    with values.open('x', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    record = {'status': 'complete_task_error_ratio_figure_created',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_metrics': str(args.metrics.resolve()), 'source_metrics_sha256': digest(args.metrics),
              'script_sha256': digest(Path(__file__)), 'complete_tasks': 12, 'cells': len(cells),
              'cell_values': 'model endpoint MSE divided by matched persistence or zero-return endpoint MSE',
              'color': 'shared logarithmic ratio scale centered at 1',
              'artifacts': {p.name: digest(p) for p in paths + [values]},
              'independent_test': False, 'Q3_values_read': False,
              'limitations': ['Exposed historical development; incomplete minute tasks are excluded explicitly.',
                              'The annual task uses the user-accepted shorter history.',
                              'A value below one describes these saved errors and is not independent significance.']}
    (args.output / 'FIGURE_AUDIT.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'status': record['status'], 'directory': str(args.output.resolve()), 'cells': len(cells)}))


if __name__ == '__main__':
    main()
