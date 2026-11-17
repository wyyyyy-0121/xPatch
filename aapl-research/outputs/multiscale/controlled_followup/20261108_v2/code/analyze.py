"""Report every selected cell; no fitting or candidate selection."""
from common import *
from datetime import datetime,timezone

def pred(path,origins,expected_sha=None):
    if expected_sha:assert sha(path)==expected_sha
    with np.load(path) as a:
        assert np.array_equal(a['origins'],origins);p=a['prediction'].astype('float64')
    assert np.isfinite(p).all()
    return p

def groups_for(bars,origins,losses):
    dates=bars.session_date[origins];grid=np.unique(bars.session_date[(bars.session_date>=dates[0])&(bars.session_date<=dates[-1])])
    ix=np.searchsorted(grid,dates);names=list(losses)
    sums=np.column_stack([np.bincount(ix,weights=losses[k],minlength=len(grid)) for k in names])
    return {'sums':sums,'counts':np.bincount(ix,minlength=len(grid)),'names':names}

def bootstrap(groups,pairs,block,task):
    names=groups[0]['names'];rng=np.random.default_rng(2026);R=5000
    if any(len(g['counts'])<=block for g in groups):
        return [{'task':task,'primary':a,'reference':b,'block':block,'status':'insufficient_groups','supports_improvement':False} for a,b in pairs]
    samples=np.zeros((R,len(names)));ns=np.zeros(R);point=np.zeros(len(names));N=0
    for g in groups:
        n=len(g['counts']);point+=g['sums'].sum(0);N+=g['counts'].sum()
        for start in range(0,R,100):
            r=min(100,R-start);starts=rng.integers(0,n-block+1,size=(r,(n+block-1)//block))
            idx=(starts[:,:,None]+np.arange(block)).reshape(r,-1)[:,:n]
            samples[start:start+r]+=g['sums'][idx].sum(1);ns[start:start+r]+=g['counts'][idx].sum(1)
    assert (ns>0).all();samples/=ns[:,None];point/=N;out=[]
    for primary,reference in pairs:
        i,j=names.index(primary),names.index(reference);delta=samples[:,j]-samples[:,i]
        varying=bool(np.any(delta!=delta[0]));ci=np.quantile(delta,[.025,.975]).tolist() if varying else None
        rel=100*(1-samples[:,i]/samples[:,j])
        out.append({'task':task,'primary':primary,'reference':reference,'block':block,'repetitions':R,
            'status':'descriptive' if varying else 'degenerate','mse_reference_minus_primary':float(point[j]-point[i]),
            'difference_ci95':ci,'relative_improvement_pct':float(100*(1-point[i]/point[j])),
            'relative_ci95_pct':np.quantile(rel,[.025,.975]).tolist() if varying else None,
            'supports_improvement':bool(ci and ci[0]>0),'scope':'exposed historical development; not multiplicity-adjusted'})
    return out

def factorial_interaction(groups,block,task):
    """Negative loss contrast supports a joint benefit beyond additive factors."""
    weights={'long_aligned':1.,'long_path':-1.,'short_aligned':-1.,'short_path':1.}
    converted=[]
    for g in groups:
        s=sum(value*g['sums'][:,g['names'].index(name)] for name,value in weights.items())
        converted.append({'names':['interaction','zero_contrast'],'counts':g['counts'],
            'sums':np.column_stack((s,np.zeros_like(s)))})
    if any(len(g['counts'])<=block for g in converted):
        return {'task':task,'block':block,'status':'insufficient_groups'}
    rng=np.random.default_rng(2026);draws=np.zeros(5000);counts=np.zeros(5000)
    total=sum(g['sums'][:,0].sum() for g in converted);N=sum(g['counts'].sum() for g in converted)
    for g in converted:
        n=len(g['counts'])
        for start in range(0,5000,100):
            r=min(100,5000-start);begins=rng.integers(0,n-block+1,size=(r,(n+block-1)//block))
            idx=(begins[:,:,None]+np.arange(block)).reshape(r,-1)[:,:n]
            draws[start:start+r]+=g['sums'][idx,0].sum(1);counts[start:start+r]+=g['counts'][idx].sum(1)
    values=draws/counts;varying=bool(np.any(values!=values[0]))
    ci=np.quantile(values,[.025,.975]).tolist() if varying else None
    return {'task':task,'block':block,'mean_loss_interaction':float(total/N),'difference_ci95':ci,
        'status':'descriptive' if varying else 'degenerate',
        'joint_gain_beyond_additivity_supported':bool(ci and ci[1]<0),
        'definition':'loss(long,aligned)-loss(long,path)-loss(short,aligned)+loss(short,path)',
        'scope':'descriptive historical contrast; three-seed per-origin mean loss'}

def main():
    assert read(BASE/'audit/TRAINING_COMPLETE.json')['status']=='all_selected_jobs_complete'
    jobs=read(BASE/'protocol/SELECTED_JOBS.json')['jobs'];rows=[];pooled=[];intervals=[];probes=[];stability=[];checks=[];interactions=[]
    for task in sorted(read(BASE/'protocol/DATA.json')['task_files']):
        meta=read(BASE/'cache'/task/'TASK.json');profile=meta['profile'];h=meta['horizon'];target=meta['target']
        bars=admitted_bars(meta['frequency']);mechanism=(profile,h,target) in MECHANISM
        variants=['short_path','short_aligned','long_path','long_aligned'] if mechanism else ['long_aligned']
        taskgroups=[];series={v:[] for v in variants};baseline_total=0.;n=0
        baseline_name='zero' if target=='future_log_return' else 'persistence'
        seed_totals={v:{} for v in variants};positive={v:0 for v in variants};fold_losses=[]
        for fn,fold in meta['folds'].items():
            with np.load(fold['targets']) as a:y=a['actual'];o=a['origins'];anchor=a['anchor']
            base=(y[:,-1] if target=='future_log_return' else y[:,-1]-anchor)**2
            losses={baseline_name:base};baseline_total+=base.sum();n+=len(y)
            for v in variants:
                selected=[j for j in jobs if key(j['profile'],j['horizon'],j['target'])==task and
                    j['family']=='tiny' and f"{j['field']}_{j['loss']}"==v and j['fold']==int(fn)]
                assert len(selected) in (1,3);lsum=np.zeros(len(y));path_sums=[];maes=[]
                for j in selected:
                    folder=BASE/'runs'/j['id'];done=read(folder/'DONE.json');assert done['status']=='completed'
                    p=pred(folder/'prediction.npz',o,done['prediction_sha256']);m=metric(p,y,anchor,target)
                    rows.append({'task':task,'variant':v,'fold':int(fn),'seed':j['seed'],**m,
                        'best_epoch':done['best_epoch'],'epochs':done['epochs'],'parameters':done['parameters'],
                        'wall_seconds':done['training_wall_seconds'],'cpu_seconds':done['training_cpu_seconds']})
                    e=(p[:,-1]-y[:,-1])**2;lsum+=e;path_sums.append(m['path_mse']*len(y));maes.append(m['endpoint_mae']*len(y))
                    seed_totals[v][j['seed']]=seed_totals[v].get(j['seed'],0.)+e.sum()
                losses[v]=lsum/len(selected)
                series[v].append({'squared_error_sum':float(losses[v].sum()),'path_error_sum':float(np.mean(path_sums)),
                    'mae_sum':float(np.mean(maes)),'origins':len(y),'seed_count':len(selected)})
                positive[v]+=int(losses[v].mean()<base.mean())
                fold_losses.append({'task':task,'variant':v,'fold':int(fn),'seed_count':len(selected),
                    'endpoint_mse':float(losses[v].mean()),'baseline_mse':float(base.mean()),
                    'improvement_pct':float(100*(1-losses[v].mean()/base.mean()))})
            if mechanism:
                rec=fold['models']['lstm'];p=pred(rec['prediction'],o,rec['prediction_sha256']);losses['original_lstm']=(p[:,-1]-y[:,-1])**2
                for name,file in [('ridge','ridge_prediction.npz'),('shrink_lstm','shrink_prediction.npz')]:
                    p=pred(BASE/'controls'/task/f'fold_{fn}'/file,o);losses[name]=(p[:,-1]-y[:,-1])**2
            g=groups_for(bars,o,losses);taskgroups.append(g)
            save_npz(BASE/'analysis/paired_losses'/task/f'fold_{fn}.npz',origins=o,session_date=bars.session_date[o],
                **{f'loss_{name}':value for name,value in losses.items()})
        for v,s in series.items():
            total=sum(a['squared_error_sum'] for a in s)
            pooled.append({'task':task,'profile':profile,'horizon':h,'target':target,'variant':v,'origins':n,
                'seed_count':s[0]['seed_count'],'completed_folds':5,'endpoint_mse':total/n,'baseline_mse':baseline_total/n,
                'endpoint_mae':sum(a['mae_sum'] for a in s)/n,'path_mse':sum(a['path_error_sum'] for a in s)/n,
                'improvement_pct':100*(1-total/baseline_total),'positive_fold_means':positive[v],
                'seed_improvements_pct':{str(k):100*(1-z/baseline_total) for k,z in seed_totals[v].items()}})
        if mechanism:
            pairs=[('long_aligned',c) for c in ['zero','original_lstm','ridge','shrink_lstm','short_path','short_aligned','long_path']]
            pairs += [('long_path','short_path'),('short_aligned','short_path')]
            for block in ([1,5] if profile=='minute' else [2,5]):
                intervals+=bootstrap(taskgroups,pairs,block,task)
                interactions.append(factorial_interaction(taskgroups,block,task))
            mainrow=next(r for r in pooled if r['task']==task and r['variant']=='long_aligned')
            relevant=[r for r in intervals if r['task']==task and r['primary']=='long_aligned' and r['reference'] in ('zero','original_lstm','ridge','shrink_lstm')]
            pchecks={'minimum_one_percent':mainrow['improvement_pct']>=1,
                'all_three_seeds_positive':len(mainrow['seed_improvements_pct'])==3 and all(x>0 for x in mainrow['seed_improvements_pct'].values()),
                'four_positive_folds':mainrow['positive_fold_means']>=4,
                'all_control_intervals_both_blocks':len(relevant)==8 and all(r['supports_improvement'] for r in relevant)}
            checks.append({'task':task,'checks':pchecks,'performance_goal_passed':all(pchecks.values())})
        else:
            # Preserve the original scale-specific block rules; pool long-scale chronology,
            # because annual fold windows can be shorter than a single 12-month block.
            pairs=[('long_aligned',baseline_name)];blocks={'annual':[12,24],'quarterly':[3,6],'monthly':[21,42],
                'hourly':[2,5],'minute':[1,5]}[profile]
            gg=taskgroups
            if profile in ('annual','quarterly','monthly'):
                gg=[{'names':gg[0]['names'],'counts':np.concatenate([g['counts'] for g in gg]),
                    'sums':np.concatenate([g['sums'] for g in gg])}]
            for block in blocks:intervals+=bootstrap(gg,pairs,block,task)
        stability+=fold_losses
        print(json.dumps({'analyzed':task}),flush=True)
    for j in jobs:
        if j['family']!='original':continue
        task=key(j['profile'],j['horizon'],j['target']);meta=read(BASE/'cache'/task/'TASK.json');rec=meta['folds'][str(j['fold'])]
        with np.load(rec['targets']) as a:y=a['actual'];o=a['origins'];anchor=a['anchor']
        done=read(BASE/'runs'/j['id']/'DONE.json');p=pred(BASE/'runs'/j['id']/'prediction.npz',o,done['prediction_sha256'])
        q=pred(rec['models'][j['model']]['prediction'],o,rec['models'][j['model']]['prediction_sha256'])
        new=float(((p[:,-1]-y[:,-1])**2).mean());old=float(((q[:,-1]-y[:,-1])**2).mean())
        probes.append({'task':task,'model':j['model'],'fold':j['fold'],'seed':2026,
            'original_endpoint_mse':old,'aligned_endpoint_mse':new,'improvement_vs_original_pct':100*(1-new/old),
            'scope':'two-fold retraining probe; same original recipe except loss'})
    fusion=analyze_fusion()
    result={'status':'complete_development_analysis','pooled':pooled,'fold_metrics':stability,
        'mechanism_performance':checks,'both_mechanism_tasks_pass':all(c['performance_goal_passed'] for c in checks),
        'intervals':intervals,'factorial_interactions':interactions,'original_loss_probes':probes,'fusion':fusion,'selected_fits':len(jobs),
        'completed_new_fits':len(rows)+len(probes),'new_primary_tasks':len([r for r in pooled if r['variant']=='long_aligned']),
        'Q3_values_read':False,'independent_confirmation':False,'at_utc':datetime.now(timezone.utc).isoformat()}
    assert result['completed_new_fits']==len(jobs) and result['new_primary_tasks']==20
    atomic(BASE/'analysis/RESULTS.json',result)
    for name,data in [('NEW_POOLED.csv',pooled),('NEW_FOLD_SEED.csv',rows),('NEW_FOLD_MEANS.csv',stability),('LOSS_PROBES.csv',probes),('PAIRED_INTERVALS.csv',intervals)]:
        fields=list(dict.fromkeys(k for d in data for k in d))
        with (BASE/'analysis'/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)

def analyze_fusion():
    results=[]
    for profile,h,target in MECHANISM:
        task=key(profile,h,target);meta=read(BASE/'cache'/task/'TASK.json');bars=admitted_bars(meta['frequency'])
        gs=[];sums={};N=0
        for fn in (3,4,5):
            rec=meta['folds'][str(fn)]
            with np.load(rec['targets']) as a:y=a['actual'];o=a['origins']
            losses={'zero':y[:,-1]**2};N+=len(y)
            for model in ('mlp','gru','lstm','xpatch'):
                p=pred(rec['models'][model]['prediction'],o);losses[model]=(p[:,-1]-y[:,-1])**2
            folder=BASE/'fusion'/task/f'fold_{fn}'
            for name in ('equal','constant'):
                p=pred(folder/f'{name}_prediction.npz',o);losses[name]=(p[:,-1]-y[:,-1])**2
            gatefiles=[folder/f'gate_s{s}_prediction.npz' for s in (2026,2027,2028)]
            if all(p.exists() for p in gatefiles):
                losses['gate']=np.mean([(pred(p,o)[:,-1]-y[:,-1])**2 for p in gatefiles],axis=0)
            for k,v in losses.items():sums[k]=sums.get(k,0.)+float(v.sum())
            gs.append(groups_for(bars,o,losses))
            save_npz(BASE/'analysis/fusion_paired_losses'/task/f'fold_{fn}.npz',origins=o,session_date=bars.session_date[o],
                **{f'loss_{name}':value for name,value in losses.items()})
        paired=[]
        if 'gate' in sums:
            for block in ([1,5] if profile=='minute' else [2,5]):
                paired+=bootstrap(gs,[('gate',x) for x in ['zero','equal','constant','mlp','gru','lstm','xpatch']],block,task)
        results.append({'task':task,'evaluation_folds':[3,4,5],'origins':N,
            'pooled':[{'method':k,'endpoint_mse':v/N,'improvement_pct':100*(1-v/sums['zero'])} for k,v in sums.items()],
            'intervals':paired,'gate_gain_supported':bool(paired) and all(p['supports_improvement'] for p in paired),
            'scope':'expert seed 2026, three gate initializations; exposed historical replay'})
    return results

if __name__=='__main__':main()
