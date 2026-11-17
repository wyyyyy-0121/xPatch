"""Independent control reconstruction from immutable paths and original targets.

No scoring runs at import. The verifier calls this only after full result notice.
"""
from pathlib import Path
import hashlib,json
import numpy as np

B=Path(__file__).resolve().parents[1]
V=B.with_name('20261109_v3')
A=B.with_name('20261109_v3a')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def source_controls(task,fold,origins,y):
    """Recompute per-origin endpoint loss; never average stochastic forecasts."""
    truth=np.asarray(y)[:,-1];losses={};identities={}
    archive=V/'analysis/candidate_paired_losses'/task/f'fold_{fold}.npz'
    identities[str(archive)]=sha(archive)
    with np.load(archive,allow_pickle=False) as a:
        assert np.array_equal(a['origins'],origins)
        assert np.array_equal(a['loss_zero'],truth**2)
        losses['zero']=a['loss_zero'].copy()
        losses['archived_default_raw']=a['loss_original'].copy()
        losses['archived_default_shrink']=a['loss_original_shrink'].copy()
        for name in ('ridge_history','ridge_history_anchor','ridge_history_selected','ridge_history_anchor_selected'):
            losses[name]=a['loss_'+name].copy()
    # Reconstruct archived single-seed defaults and every Ridge control from
    #complete saved forecast paths as well as checking the loss-file lineage.
    control=V/'controls_cross_scale'/task/f'fold_{fold}'
    for filename,receipt_name,columns in [
        ('original_predictions.npz','ORIGINAL_SHRINK.json',{'archived_default_raw':'original','archived_default_shrink':'shrink'}),
        ('predictions.npz','DONE.json',{'ridge_history':'history','ridge_history_anchor':'history_anchor','ridge_history_selected':'history_selected','ridge_history_anchor_selected':'history_anchor_selected'})]:
        receipt=read(control/receipt_name);file=control/filename
        assert sha(file)==receipt['prediction_sha256'];identities[str(file)]=sha(file)
        with np.load(file,allow_pickle=False) as a:
            assert np.array_equal(a['origins'],origins)
            for dest,src in columns.items():
                actual=(a[src][:,-1]-truth)**2
                assert np.array_equal(losses[dest],actual),dest
    for label,variant,calibration in [('default','anchor_random','calibration'),('repair','anchor_baseline','calibration_candidate')]:
        per_seed={'raw':[],'shrink':[]}
        for seed in (2026,2027,2028):
            job=f'{task}_xpatch_{variant}_f{fold}_s{seed}'
            run=V/'runs'/job;done=read(run/'DONE.json')
            file=run/'predictions.npz';assert sha(file)==done['prediction_sha256'];identities[str(file)]=sha(file)
            with np.load(file,allow_pickle=False) as a:
                assert np.array_equal(a['origins'],origins);raw=a['trained'].copy()
            cal=V/calibration/job;receipt=read(cal/'DONE.json')
            assert receipt['source_checkpoint_sha256']==done['checkpoint_sha256'] and not receipt['outer_labels_used_for_calibration']
            cfile=cal/'predictions.npz';assert sha(cfile)==receipt['prediction_sha256'];identities[str(cfile)]=sha(cfile)
            with np.load(cfile,allow_pickle=False) as a:
                assert np.array_equal(a['origins'],origins)
                shrink=receipt['alpha']*raw
                assert np.allclose(a['prediction'],shrink,rtol=1e-13,atol=1e-15)
            for mode,p in [('raw',raw),('shrink',shrink)]:
                error=(p[:,-1]-truth)**2
                losses[f'{label}_seed_{seed}_{mode}']=error
                per_seed[mode].append(error)
        for mode,errors in per_seed.items():losses[f'{label}_three_seed_{mode}']=np.mean(errors,axis=0)
    mfile=A/'mean_controls/predictions'/task/f'fold_{fold}.npz'
    mrec=read(A/'mean_controls/fits'/task/f'fold_{fold}'/'DONE.json')
    assert sha(mfile)==mrec['prediction_sha256'];identities[str(mfile)]=sha(mfile)
    with np.load(mfile,allow_pickle=False) as a:
        assert np.array_equal(a['origins'],origins) and np.array_equal(a['loss_baseline'],truth**2)
        for mode in ('raw','shrink'):
            error=(a[mode][:,-1]-truth)**2
            assert np.array_equal(error,a['loss_'+mode])
            losses['fit_mean_'+mode]=error
    return losses,identities
