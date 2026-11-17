"""Publication figures from complete frozen result tables, without model fitting."""
from study import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    assert len(os.sched_getaffinity(0)) <= 2
    rep = read(BASE/'analysis/REPLICATION_RESULTS.json')
    candidate = read(BASE/'analysis/CANDIDATE_RESULTS.json'); cap = read(BASE/'analysis/CAP_RESULTS.json')
    out = BASE/'manuscript/figures'; out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9, 'axes.spines.right': False,
                         'axes.spines.top': False, 'savefig.dpi': 240, 'axes.titlesize': 10})
    fig, axes = plt.subplots(2, 1, figsize=(4.3, 4.5), constrained_layout=True)
    methods = ['anchor_random__trained', 'noanchor_random__trained', 'anchor_baseline__trained',
               'noanchor_baseline__trained', 'anchor_baseline__shrink']
    labels = ['Default', 'No anchor', 'Baseline init.', 'Both', 'Init. + shrink']
    colors = ['#817777', '#6596ad', '#4b7b71', '#91ad91', '#2c4976']
    for ax, task, title in zip(axes, ('minute_h60_future_log_return', 'hourly_h8_future_log_return'),
                               ('Minute H60 return', 'Hourly H8 return')):
        selected = [next(r for r in rep['pooled'] if r['task'] == task and r['model'] == 'xpatch' and r['method'] == m)
                    for m in methods]
        ratios = [r['endpoint_mse']/r['baseline_mse'] for r in selected]
        ax.barh(np.arange(len(methods)), ratios, color=colors, height=.64)
        ax.axvline(1., color='black', linewidth=.9, linestyle='--', label='Zero return')
        ax.set_yticks(np.arange(len(methods)), labels); ax.invert_yaxis()
        ax.set_title(title); ax.set_xlim(0, max(ratios)*1.17)
        ax.set_xlabel('Endpoint MSE / zero-return MSE')
        for i, value in enumerate(ratios):
            ax.text(value+.015, i, f'{value:.4f}', va='center', fontsize=8)
        ax.grid(axis='x', alpha=.16); ax.set_axisbelow(True)
    fig.savefig(out/'return_controls.png'); fig.savefig(out/'return_controls.pdf'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.7, 3.6), constrained_layout=True)
    names = ['default20', 'default60', 'price_baseline20', 'residual_baseline20']
    xs = np.arange(4); width=.34
    for i, (model, color) in enumerate((('lstm', '#6596ad'), ('xpatch', '#4b7b71'))):
        values = [next(r['endpoint_mse'] for r in cap['pooled'] if r['model'] == model and r['method'] == name)
                  for name in names]
        ax.bar(xs+(i-.5)*width, values, width, color=color, label=model.upper() if model == 'lstm' else 'xPatch')
    baseline = next(r['endpoint_mse'] for r in cap['pooled'] if r['method'] == 'persistence')
    ax.axhline(baseline, color='black', linestyle='--', linewidth=.9, label='Persistence')
    ax.set_yscale('log'); ax.set_ylabel('Hourly-H1 price MSE (dollars squared, log scale)')
    ax.set_xticks(xs, ['Default / 20', 'Default / 60', 'Price init. / 20', 'Residual init. / 20'])
    ax.legend(frameon=False, fontsize=8); ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    fig.savefig(out/'cap_diagnostic.png'); fig.savefig(out/'cap_diagnostic.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.0), constrained_layout=True, sharey=True)
    for ax, target, title in zip(axes, ('future_close', 'future_log_return'), ('Nominal price', 'Cumulative return')):
        rows = [next(r for r in candidate['pooled'] if r['task'] == task_key(p, h, target) and r['method'] == 'candidate')
                for p, (_, _, hs) in PROFILES.items() for h in hs]
        vals = [r['improvement_pct'] for r in rows]
        ax.barh(np.arange(10), vals, color=['#4b7b71' if x > 0 else '#a26565' for x in vals])
        ax.axvline(0, color='black', linewidth=.8); ax.set_title(title)
        ax.set_xlabel('MSE improvement over persistence / zero (%)')
        ax.grid(axis='x', alpha=.18); ax.set_axisbelow(True)
    labels = [f'{p.capitalize()} H{h}' for p, (_, _, hs) in PROFILES.items() for h in hs]
    axes[0].set_yticks(np.arange(10), labels); axes[0].invert_yaxis()
    fig.savefig(out/'candidate_twenty_tasks.png'); fig.savefig(out/'candidate_twenty_tasks.pdf'); plt.close(fig)
    atomic(BASE/'audit/FIGURES.json', {'status': 'completed', 'figures': 3,
           'source_results_sha256': sha(BASE/'analysis/CANDIDATE_RESULTS.json'),
           'files': {p.name: sha(p) for p in out.iterdir() if p.suffix in ('.png', '.pdf')}, 'at_utc': now()})


if __name__ == '__main__':
    main()
