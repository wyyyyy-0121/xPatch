from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {'zero':'Zero', 'lstm':'LSTM', 'trend':'Trend', 'short_plain':'Local',
    'short_const':'Local + constant', 'short_state':'Local + state',
    'linear_same_features':'Ridge (488)', 'lstm_constant_shrink':'LSTM + scalar',
    'short':'Local', 'fixed_mix':'Fixed mixture', 'mix_noshrink':'State mixture', 'full':'Full model'}


def main():
    results = read(BASE/'analysis/RESULTS.json'); diagnostic = read(BASE/'diagnostics/DIAGNOSTICS.json')
    out = BASE/'manuscript/figures'; out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':8, 'axes.spines.top':False, 'axes.spines.right':False,
        'figure.dpi':160, 'savefig.dpi':300, 'font.family':'DejaVu Sans'})
    def save(fig, name):
        fig.savefig(out/f'{name}.png', bbox_inches='tight')
        fig.savefig(out/f'{name}.pdf', bbox_inches='tight')
        plt.close(fig)
    models = diagnostic['models']; yy = np.arange(4)
    fig, ax = plt.subplots(figsize=(3.4,2.55))
    energy = [100*diagnostic['pooled'][m]['energy_ratio'] for m in models]
    alignment = [100*diagnostic['pooled'][m]['alignment_ratio'] for m in models]
    ax.barh(yy+.16, energy, height=.30, label=r'$E[\hat y^2]/E[y^2]$', color='#bb4a42')
    ax.barh(yy-.16, alignment, height=.30, label=r'$2E[\hat y y]/E[y^2]$', color='#2e6f9e')
    ax.set_yticks(yy, ['MLP','GRU','LSTM','xPatch']); ax.invert_yaxis()
    ax.set_xscale('log'); ax.set_xlim(.15,100)
    ax.set_xlabel('Percent of zero-reference MSE (log scale)')
    ax.legend(loc='upper right', fontsize=7, frameon=False)
    fig.tight_layout(); save(fig, 'diagnostic_energy')
    kinds = [r['kind'] for r in results['pooled'] if r['kind'] != 'zero']
    fig, ax = plt.subplots(figsize=(3.4,3.2))
    colors = ['#2e6f9e','#e89042','#59a775','#9a6db0','#bb4a42']
    for i, kind in enumerate(kinds):
        for fold in range(1,6):
            vals = [r['h60_improvement_pct'] for r in results['fold_seed_metrics']
                    if r['kind']==kind and r['fold']==fold]
            ax.scatter(vals, np.full(len(vals), i)+(fold-3)*.085, s=10,
                       color=colors[fold-1], alpha=.85, label=f'Fold {fold}' if i==0 else None)
        pooled = next(r for r in results['pooled'] if r['kind']==kind)
        ax.scatter(pooled['h60_improvement_pct'], i, marker='D', s=22, color='black', zorder=4)
    ax.axvline(0, color='black', lw=.8); ax.axvline(1, color='gray', ls=':', lw=.8)
    ax.set_yticks(np.arange(len(kinds)), [LABELS[k] for k in kinds]); ax.invert_yaxis()
    ax.set_xlabel('H60 MSE improvement over zero (%)')
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),fontsize=6.5,frameon=False,ncol=3)
    fig.tight_layout(); save(fig, 'formal_stability')
    fig, ax = plt.subplots(figsize=(6.4,3.0))
    zero = next(r for r in results['pooled'] if r['kind']=='zero')
    for row in results['pooled']:
        if row['kind']=='zero': continue
        ratio = np.array(row['per_step_mse'])/np.array(zero['per_step_mse'])
        ax.plot(np.arange(1,61), ratio, label=LABELS[row['kind']], lw=1.1)
    ax.axhline(1, color='black', ls=':', lw=.8); ax.set_xlabel('Cumulative-return endpoint (trading minutes)')
    ax.set_ylabel('MSE / zero-reference MSE'); ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout(); save(fig, 'auxiliary_path')


if __name__=='__main__': main()
