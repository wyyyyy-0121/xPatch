"""Exercise a real tiny four-expert / three-seed export, without any foundation model."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from .common import OUT,PLAN,digest,write_json,source_identity
from .tests_data import two_sessions
from .tests_training import make_panel
from .experts import fit_cpu,tabular
from .experiment import model_for
from .trainer import Trainer
from .inference import export,FrozenSystem


def test_real_fitted_four_experts_and_three_seeds_export_load_exactly(monkeypatch):
    frame,starts,ends=two_sessions();panel=make_panel(frame,starts,ends)
    root=OUT/'deployment';root.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='unit_ensemble_',dir=root) as folder:
        parent=Path(folder);references={};systems=[];in_memory={};scalers={}
        for family in ('ridge','lightgbm'):
            config={'family':family,'lookback':48,'alpha':1000.,'rows_per_day':8,'trees':4,'num_leaves':7,
                    'max_depth':3,'min_child_samples':4,'learning_rate':.03,'reg_lambda':10.,'colsample_bytree':1.,'cpu_threads':2}
            prediction,scale,_=fit_cpu(panel,config,'2021-01-04',parent/family)
            references[family]={'config':config,'directory':str(parent/family)}
            in_memory[family]=prediction;scalers[family]=scale
        for family in ('xpatch','patchtst'):
            config={'family':family,'lookback':48,'lr':.001,'batch_size':4,'total_updates':2,'loss':'raw'}
            model=model_for(panel,config,'cpu')
            trainer=Trainer(panel,model,config,'2021-01-04',parent.name+'/'+family,device='cpu')
            trainer.run_until(2);model.eval()
            references[family]={'config':config,'checkpoint':str(trainer.path)}
            in_memory[family]=model;scalers[family]=trainer.scalers
        cfg={'family':'fusion','lookback':48,'hidden':32,'projection':64,'market_encoder':'tcn','residual_cap':.25,
             'foundation':False,'market':True,'lr':.001,'batch_size':4,'total_updates':2,'loss':'raw'}
        class ToyOOF:
            available={0:np.zeros(len(panel.dates),dtype=bool)}
            sources={'test':'no early in-sample expert predictions'}
        trained=[]
        for seed in (2026,2027,2028):
            config={**cfg,'seed':seed};torch.manual_seed(seed);model=model_for(panel,config,'cpu')
            trainer=Trainer(panel,model,config,'2021-01-04',parent.name+'/system',seed,ToyOOF(),device='cpu')
            trainer.run_until(2);model.eval();trained.append((model,trainer.scalers,config))
            systems.append({'config':config,'checkpoint':str(trainer.path)})
        evidence=parent/'evidence';freeze=evidence/'system_freeze.json'
        write_json(freeze,{'source_identity':source_identity(),'plan_sha256':digest(PLAN),'unit_test':True})
        destination=parent/'package';manifest=export(destination,panel.stocks,panel.background,references,systems,digest(freeze))
        loaded=FrozenSystem(destination,'cpu');rows=np.array([430,500,700]);stock=np.zeros(3,dtype=np.int64)
        experts=[]
        with torch.no_grad():
            for family in ('ridge','lightgbm','xpatch','patchtst'):
                batch=panel.windows(stock,rows,references[family]['config'],scalers[family],include_labels=False)
                values=in_memory[family](tabular(batch,len(panel.stocks))) if family in ('ridge','lightgbm') else in_memory[family](batch)['prediction'].numpy()
                experts.append(values)
            experts=np.asarray(experts,dtype=np.float32).T;expected=[]
            for model,scale,config in trained:
                batch=panel.windows(stock,rows,config,scale,experts,include_labels=False)
                expected.append(model(batch)['prediction'].numpy())
        output=loaded.predict_origin(panel,rows)
        np.testing.assert_array_equal(output['expert_predictions'],experts)
        np.testing.assert_array_equal(output['seed_predictions'],np.stack(expected))
        from .evaluation import seed_system
        _,development_ensemble=seed_system(panel.targets['AAPL'][rows],np.stack(expected))
        np.testing.assert_array_equal(output['prediction'],development_ensemble)
        assert loaded.model_identity==manifest['identity']
        # Execute the same exported fits after isolated packaging, without any
        # development arrays or old checkout. All example values are fabricated.
        import pandas as pd
        from .stream import Stream
        from . import local_package
        shifted=frame.copy();shifted.index=shifted.index+(pd.Timestamp('2026-01-05')-pd.Timestamp('2021-01-04'))
        acceptance=evidence/'deployment_acceptance'/manifest['identity']
        stream=Stream(shifted.index,panel.stocks,panel.background,cfg,trained[0][1],loaded,manifest['identity'],acceptance/'state.json')
        origin=350
        for stamp,row in shifted.iloc[:origin+1].iterrows():
            for name in stream.names:stream.ingest(name,stamp,row.to_dict(),stamp+pd.Timedelta(minutes=16))
        assert stream.predict(shifted.index[origin],shifted.index[origin]+pd.Timedelta(minutes=16))['status']=='ok'
        write_json(acceptance/'report.json',{'unit_test':True,'model_identity':manifest['identity'],'Q3_accessed':False,'engineering_acceptance_passed':False})
        write_json(acceptance/'progress.json',{'recipe':{'source_identity':source_identity(),'system_freeze_sha256':digest(freeze)}})
        write_json(evidence/'confirmation.json',{'unit_test':True,'predictive_success':False})
        monkeypatch.setattr(local_package,'OUT',evidence)
        archive=local_package.build(destination);portable=local_package.verify(archive)
        assert portable['isolated_example_and_fresh_process_resume_passed'] and portable['files_hash_verified']>50
        # Temporary training checkpoints are test artifacts, not research results.
        import shutil
        shutil.rmtree(OUT/'training'/parent.name)
