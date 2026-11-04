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
    fig,axes = plt.subplots(2,1,figsize=(7.2,3.9),sharex=True,sharey=True)
    fig.subplots_adjust(left=.105,right=.98,top=.85,bottom=.16,hspace=.38)
    for ax,panel in zip(axes,panels):
        for name in ('ridge','lightgbm','xpatch','chronos2'):
            values = result['comparisons'][f'{panel}/{name}']['quarter_differences']
            if sorted(values)!=quarters:
                raise ValueError('Quarter coverage differs between candidates')
            ax.plot(range(10),[values[q] for q in quarters],marker='o',markersize=4,
                color=colors[name],linewidth=1.6,label={'ridge':'Ridge','lightgbm':'LightGBM',
                    'xpatch':'xPatch','chronos2':'Chronos-2'}[name])
        ax.axhline(0,color='#262F36',linewidth=.9,linestyle='--')
        ax.set_ylabel('QLIKE difference')
        ax.set_ylim(-.05,.065)
        ax.set_title('AAPL history' if panel=='aapl' else 'AAPL + SPY + QQQ history',fontsize=9,loc='left')
        ax.grid(axis='y',alpha=.2)
    axes[-1].set_xticks(range(10),quarters,rotation=35,ha='right')
    handles,legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.54,.995),ncol=4,frameon=False)
    save(fig,'quarterly_loss_differences')
    manifest = {'source':str(SOURCE.relative_to(ROOT)),'source_sha256':digest(SOURCE),
        'script':str(Path(__file__).relative_to(ROOT)),'script_sha256':digest(Path(__file__)),
        'matplotlib_version':matplotlib.__version__,'numpy_version':np.__version__,
        'scope':'static_visualization_of_existing_results_no_new_scores_or_tests',
        'historical_exposure':True,'quarter_count':10,'information_sets':list(panels),
        'files':{str(p.relative_to(ROOT)):digest(p) for p in files}}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':True,'figures':2,'files':len(files),'directory':str(OUTPUT)}))


if __name__ == '__main__':
    main()
