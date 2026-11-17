from common import *
from datetime import datetime,timezone
import time

def make_job(profile,h,target,fold,seed,family='tiny',model='tiny',field='long',loss='aligned',phase='formal'):
    freq,lookback,_=PROFILES[profile]
    name=(f'{model}_aligned' if family=='original' else f'{field}_{loss}')
    return {'id':f'{key(profile,h,target)}_{name}_f{fold}_s{seed}',
            'profile':profile,'frequency':freq,'horizon':h,'lookback':lookback,'target':target,
            'fold':fold,'seed':seed,'family':family,'model':model,'field':field,'loss':loss,'phase':phase}

def main():
    assert read(BASE/'protocol/DATA.json')['status']=='matched_original_twenty_tasks'
    jobs=[]
    for profile,h,target in MECHANISM:
        for model in ('mlp','gru','lstm','xpatch'):
            for fold in (1,2):jobs.append(make_job(profile,h,target,fold,2026,'original',model,phase='loss_probe'))
        for field in ('short','long'):
            for loss in ('path','aligned'):
                for fold in range(1,6):
                    for seed in (2026,2027,2028):
                        jobs.append(make_job(profile,h,target,fold,seed,field=field,loss=loss,
                            phase='mechanism_initial' if fold<=2 and seed==2026 else 'mechanism_formal'))
    for profile,(_,_,horizons) in PROFILES.items():
        for h in horizons:
            for target in ('future_close','future_log_return'):
                if (profile,h,target) in MECHANISM:continue
                for fold in range(1,6):
                    for seed in (2026,2027,2028):jobs.append(make_job(profile,h,target,fold,seed,phase='cross_scale'))
    assert len(jobs)==406 and len({j['id'] for j in jobs})==406
    atomic(BASE/'protocol/ALL_JOBS.json',{'status':'matrix_before_new_scores','jobs':jobs})
    atomic(BASE/'protocol/CODE_FREEZE.json',{'status':'frozen_before_training','hashes':code_hashes(),
        'data_sha256':sha(BASE/'protocol/DATA.json'),'at_utc':datetime.now(timezone.utc).isoformat()})
    if not (BASE/'protocol/BUDGET.json').exists():
        now=time.time()
        atomic(BASE/'protocol/BUDGET.json',{'start_epoch':now,'deadline_epoch':now+8*3600,
            'maximum_wall_seconds':8*3600,'maximum_workers':8,'cpus_per_worker':2,
            'analysis_and_light_calibration_cpus_total':2,'gpu':False,'maximum_new_neural_fits':406,
            'basis':'User explicitly requested plan then immediate execution and activation of this paper goal on 2026-11-08; execute prior reviewed eight-hour proposal. Deadline never resets on resume.',
            'independent_of_terminated_foundation_project':True})
    atomic(BASE/'protocol/PREDECLARATION.json',{
        'primary':'long_aligned','mechanism_tasks':MECHANISM,'seeds':[2026,2027,2028],'folds':[1,2,3,4,5],
        'original_title':'Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales',
        'new_loss':'.5 standardized full-path MSE + .5 standardized endpoint MSE',
        'tiny':{'layers':8,'width':8,'kernel':3,'dilations_short':[1]*8,
                'dilations_long':[2**i for i in range(8)],'parameters_formula':'1432+17H',
                'input':'past split-aware log-close history demeaned within window; global scalar normalization inner fit only',
                'output':'separately fitted nominal-price residual or cumulative log return',
                'zero_residual_initialization':True,'epoch_zero_option':True},
        'training':{'adam_lr':.001,'batch_size':128,'max_epochs':20,'patience':3},
        'matrix_admission':'406 or 226 selected solely from measured throughput/memory before new market scores',
        'success':{'minimum_mse_improvement_over_zero_pct':1,'all_seeds_positive':True,'positive_fold_means_at_least':4,
            'must_beat':['zero','original_lstm_2026','same_input_ridge','inner_shrunk_original_lstm_2026'],
            'session_blocks':{'minute':[1,5],'hourly':[2,5]},'bootstrap_repetitions':5000,'bootstrap_seed':2026,
            'nominal_ci':.95,'all_control_improvement_lower_bounds_positive_for_both_blocks':True},
        'fusion':{'tasks':MECHANISM,'expert_seed':2026,'evaluation_folds':[3,4,5],
            'experts':['mlp','gru','lstm','xpatch','zero'],'gate_inputs':4,'gate_parameters':25,
            'short_long_windows':{'minute':[60,240],'hourly':[6,48]},'epsilon':1e-8,
            'feature_definitions':['v_short/max(v_long,eps)','abs(sum(r_short))/max(sqrt(short)*v_long,eps)',
                'sum(r_long)/max(sqrt(long)*v_long,eps)','known elapsed session minutes / known session length'],
            'v':'sqrt(mean(r^2)) on split-aware past adjacent log closes',
            'past_inner_split':'earliest 80% completed calibration origins fit, last 20% stop, purge fit labels overlapping first stop decision',
            'minimum_fit_stop_origins':[100,30],'lr':.01,'max_steps':300,'patience':20,
            'equal_weight_penalty':.001,'objective':'endpoint squared error / fit-label variance + penalty',
            'gate_seeds':[2026,2027,2028],'all_label_completion_strictly_before_evaluation_start':True},
        'scope':'exposed historical development, not independent confirmation',
        'Q3_values_read':False,'new_title_authorized':False,'plan_sha256':sha(BASE/'protocol/REVIEWED_PLAN_ZH.md')})

if __name__=='__main__':main()
