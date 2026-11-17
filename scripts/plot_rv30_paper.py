"""Render static manuscript figures from existing saved evaluation results.

Use a separate plotting runtime; the signed experiment environment is unchanged.
This script neither fits a model nor recomputes statistical tests.
"""
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'research/paper/figures'
SOURCE = ROOT / 'artifacts/rv30/65a4daf1ebacb8a7dedca5e866010bd94cda25acc7124509ec81d1c14dc2fbf8/development/evaluation.json'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/xpatch-paper-mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    result = json.loads(SOURCE.read_text())
    if result['errors'] or not result['historical_exposure']:
        raise ValueError('Do not plot unresolved results or drop historical-exposure disclosure')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    panels = ('aapl','market')
    names = ('last_rv','ridge','lightgbm','xpatch','chronos2')
    labels = ('Previous RV','Ridge','LightGBM','xPatch','Chronos-2')
    colors = {'ridge':'#137F79','lightgbm':'#C37C1C','xpatch':'#7460A8','chronos2':'#B84B57'}
    files = []
    def save(fig, name):
        for suffix in ('png','pdf','svg'):
            path = OUTPUT / f'{name}.{suffix}'
            fig.savefig(path, dpi=240)
            files.append(path)
        plt.close(fig)

    # Schematic only: no data-derived publication timestamps are implied.
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.35))
    fig.subplots_adjust(left=.08, right=.98, top=.91, bottom=.06, hspace=.62)
    ax = axes[0]
    ax.set_xlim(-4, 32)
    ax.set_ylim(-.65, 1.8)
    ax.axis('off')
    ax.set_title('(a) One forecast origin: time measured in minutes', loc='left', fontsize=9)
    ax.annotate('', xy=(31.5, 0), xytext=(-3.7, 0),
                arrowprops={'arrowstyle': '->', 'color': '#344658'})
    ax.plot([-3.6, -1], [0, 0], color='#137F79', linewidth=7, solid_capstyle='butt')
    ax.plot([-1, 0], [0, 0], color='#D5A33B', linewidth=7, solid_capstyle='butt')
    ax.plot([0, 30], [0, 0], color='#6688BA', linewidth=7, solid_capstyle='butt')
    for value, label in [(-1, 'Input cutoff\nu - 1'), (0, 'Origin\nu'), (30, 'Target end\nu + 30')]:
        ax.plot([value, value], [-.12, .14], color='#344658', linewidth=.7)
        ax.text(value, -.25, label, ha='right' if value == -1 else 'left' if value == 0 else 'center', va='top', fontsize=8)
    ax.text(15, .45, 'Future RV30: sum of 30 within-session squared returns', ha='center', fontsize=8)
    ax.text(-3.5, 1.1, 'History through u - 1; latest bar starts at u - 2', fontsize=8)
    ax.text(16, 1.1, 'Example: origin 09:45; target ends 10:15 ET', fontsize=8)

    ax = axes[1]
    ax.set_xlim(0, 100)
    ax.set_ylim(-.2, 1.8)
    ax.axis('off')
    ax.set_title('(b) Outer evaluation: earlier dates on the left (not to scale)', loc='left', fontsize=9)
    from matplotlib.patches import Rectangle
    for start, width, label, color in [(0, 38, 'Fitting history', '#D5EAE7'),
        (38, 9, 'Purge\n1 session', '#F2DFC0'), (47, 22, 'Calibration\n20 sessions', '#DCE5F1'),
        (69, 9, 'Purge\n1 session', '#F2DFC0'), (78, 22, 'Evaluation\nquarter', '#E8DEEF')]:
        ax.add_patch(Rectangle((start, .1), width, .7, facecolor=color, edgecolor='white'))
        ax.text(start + width/2, .45, label, ha='center', va='center', fontsize=7.5)
    ax.text(1, 1.25, 'Three expanding inner folds select parameters using earlier history only', fontsize=8)
    save(fig, 'task_and_validation')

    fig, axes = plt.subplots(1,2,figsize=(7.2,3.2),sharex=True,sharey=True)
    fig.subplots_adjust(left=.145,right=.98,top=.88,bottom=.23,wspace=.17)
    for ax,panel in zip(axes,panels):
        for y,name in enumerate(names):
            comparison = result['comparisons'][f'{panel}/{name}']
            estimate = comparison['mean_difference']
            bounds = next(b['ci95'] for b in comparison['bootstraps'] if b['block_sessions']==5)
            if not bounds[0] <= estimate <= bounds[1]:
                raise ValueError('Unexpected bootstrap interval ordering')
            color = '#137F79' if comparison['passed'] else '#647382'
            ax.errorbar(estimate,y,xerr=[[estimate-bounds[0]],[bounds[1]-estimate]],
                fmt='o',color=color,capsize=3,elinewidth=1.3,markersize=4.5)
        ax.axvline(0,color='#262F36',linewidth=.9,linestyle='--')
        ax.set_title('AAPL history' if panel=='aapl' else 'AAPL + SPY + QQQ history',pad=9,fontsize=9)
        ax.set_yticks(np.arange(len(labels)),labels)
        ax.set_ylim(len(labels)-.5,-.5)
        ax.set_xlim(-.024,.041)
        ax.set_xlabel('Mean daily QLIKE difference\n(candidate minus HAR)')
        ax.grid(axis='x',alpha=.2)
    save(fig,'development_loss_differences')

    quarters = sorted(result['comparisons']['aapl/ridge']['quarter_differences'])
    if len(quarters)!=10:
        raise ValueError('All ten quarters must be present in the manuscript figure')
    fig,axes = plt.subplots(2,1,figsize=(7.2,3.65),sharex=True,sharey=True)
    fig.subplots_adjust(left=.14,right=.87,top=.91,bottom=.15,hspace=.5)
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-.05, vcenter=0, vmax=.065)
    for ax,panel in zip(axes,panels):
        candidates = ('ridge','lightgbm','xpatch','chronos2')
        matrix = []
        for name in candidates:
            values = result['comparisons'][f'{panel}/{name}']['quarter_differences']
            if sorted(values)!=quarters:
                raise ValueError('Quarter coverage differs between candidates')
            matrix.append([values[q] for q in quarters])
        heat = ax.imshow(matrix, cmap='RdBu_r', norm=norm, aspect='auto')
        ax.set_yticks(range(4), ['Ridge','LightGBM','xPatch','Chronos-2'])
        for row in range(4):
            for col in range(10):
                value = matrix[row][col]
                ax.text(col,row,f'{value:.3f}',ha='center',va='center',fontsize=6,
                        color='white' if abs(value)>.035 else '#1F2933')
        ax.set_title('AAPL history' if panel=='aapl' else 'AAPL + SPY + QQQ history',fontsize=9,loc='left')
    axes[-1].set_xticks(range(10),quarters,rotation=35,ha='right')
    cax=fig.add_axes([.9,.23,.016,.55])
    colorbar=fig.colorbar(heat,cax=cax)
    colorbar.set_label('QLIKE difference: candidate - HAR',fontsize=7)
    save(fig,'quarterly_loss_differences')
    manifest = {'source':str(SOURCE.relative_to(ROOT)),'source_sha256':digest(SOURCE),
        'script':str(Path(__file__).relative_to(ROOT)),'script_sha256':digest(Path(__file__)),
        'matplotlib_version':matplotlib.__version__,'numpy_version':np.__version__,
        'scope':'static_visualization_of_existing_results_no_new_scores_or_tests',
        'historical_exposure':True,'quarter_count':10,'information_sets':list(panels),
        'files':{str(p.relative_to(ROOT)):digest(p) for p in files}}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':True,'figures':3,'files':len(files),'directory':str(OUTPUT)}))


if __name__ == '__main__':
    main()
