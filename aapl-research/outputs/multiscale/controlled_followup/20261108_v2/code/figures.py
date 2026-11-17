from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def save(fig,name):
    folder=BASE/'manuscript/figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/f'{name}.png',dpi=300,bbox_inches='tight')
    fig.savefig(folder/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

def main():
    x=read(BASE/'analysis/RESULTS.json');plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8})
    fig,axes=plt.subplots(1,2,figsize=(3.35,3.5))
    labels=[];matrices=[]
    for target in ('future_close','future_log_return'):
        rows=[next(r for r in x['pooled'] if r['task']==key(p,h,target) and r['variant']=='long_aligned')
            for p,(_,_,hs) in PROFILES.items() for h in hs]
        matrices.append(np.array([r['endpoint_mse']/r['baseline_mse'] for r in rows]))
        labels=[f"{r['profile'].capitalize()} {r['horizon']}" for r in rows]
    lo=min(.5,float(min(m.min() for m in matrices)));hi=max(2.,float(max(m.max() for m in matrices)))
    norm=colors.TwoSlopeNorm(vmin=np.log10(lo),vcenter=0,vmax=np.log10(hi))
    for ax,vals,title in zip(axes,matrices,['Nominal price','Log return']):
        ax.imshow(np.log10(vals[:,None]),cmap='RdBu_r',norm=norm,aspect='auto')
        ax.set_xticks([]);ax.set_yticks(range(len(labels)));ax.set_yticklabels(labels if ax is axes[0] else [])
        ax.set_title(title,fontsize=8)
        for i,v in enumerate(vals):
            tone=float(norm(np.log10(v)))
            ax.text(0,i,f'{v:.5f}',ha='center',va='center',fontsize=7,
                color='white' if tone<.22 or tone>.78 else 'black')
    fig.subplots_adjust(left=.33,wspace=.1);save(fig,'new_twenty_task_ratios')
    fig,axes=plt.subplots(1,2,figsize=(3.35,2.4),sharey=False)
    for ax,(profile,h,target) in zip(axes,MECHANISM):
        task=key(profile,h,target);variants=['short_path','long_path','short_aligned','long_aligned']
        for i,v in enumerate(variants):
            rows=[r for r in x['fold_metrics'] if r['task']==task and r['variant']==v]
            ys=[r['improvement_pct'] for r in rows]
            ax.scatter(np.full(len(ys),i),ys,s=10,color='0.55')
            pooled=next(r for r in x['pooled'] if r['task']==task and r['variant']==v)
            ax.scatter(i,pooled['improvement_pct'],marker='D',s=18,color='black')
        ax.axhline(0,color='black',lw=.6);ax.set_xticks(range(4),['S/P','L/P','S/A','L/A'],rotation=45)
        ax.set_title(f'{profile.capitalize()} H{h}',fontsize=8)
        ax.set_ylabel('MSE improvement over zero (%)',fontsize=7);ax.grid(axis='y',alpha=.2)
    fig.tight_layout(pad=.5);save(fig,'factorial_folds')
    fig,axes=plt.subplots(1,2,figsize=(3.35,2.1))
    for ax,result in zip(axes,x['fusion']):
        selected=[next(r for r in result['pooled'] if r['method']==name) for name in ['equal','constant','gate']
            if any(r['method']==name for r in result['pooled'])]
        ax.bar(range(len(selected)),[r['improvement_pct'] for r in selected],color=['.7','.45','.2'][:len(selected)])
        ax.axhline(0,color='black',lw=.6);ax.set_xticks(range(len(selected)),[r['method'] for r in selected],rotation=30)
        ax.set_title(result['task'].split('_h')[0].capitalize(),fontsize=8)
        ax.set_ylabel('MSE improvement (%)',fontsize=7)
    fig.tight_layout(pad=.5);save(fig,'fusion_replay')

if __name__=='__main__':main()
