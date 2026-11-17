from common import *
from datetime import datetime,timezone
from models import PROBES

def main():
    preflight=read(BASE/'audit/PREFLIGHT.json');assert preflight['status']=='passed'
    jobs=[{'id':f'probe_{k}_f{f}_s2026','phase':'probe','kind':k,'fold':f,'seed':2026}
          for k in PROBES for f in (1,2)]
    atomic(BASE/'protocol/JOBS.json',{'jobs':jobs})
    atomic(BASE/'protocol/CODE_FREEZE.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'hashes':code_hashes()})
    now=time_now=__import__('time').time()
    atomic(BASE/'protocol/BUDGET.json',{'start_epoch':now,'deadline_epoch':now+86400,'maximum_wall_seconds':86400,
        'maximum_workers':8,'cpus_per_worker':2,'maximum_market_fits':102,'gpu':False,
        'basis':'User activated goal to execute the reviewed follow-up plan; separate from original core budget'})
    atomic(BASE/'protocol/PREDECLARATION.json',{'status':'frozen_before_new_market_fit',
      'at_utc':datetime.now(timezone.utc).isoformat(),'primary_horizon':60,'lookback':480,'target':'future_log_return',
      'seeds':[2026,2027,2028],'folds':[1,2,3,4,5],'probe_folds':[1,2],
      'optimizer':{'adam_lr':.001,'batch_size':128,'max_epochs':20,'patience':3},
      'candidate_parameters':preflight['small_parameters'],
      'complementarity_gate':'trend-only versus residual-only xPatch losses must reverse rank across fit-defined low/high volatility states, with the same reversal orientation in both probe folds; diagnostic only',
      'shrinkage_basis':'prediction energy exceeds alignment in the already-exposed original H60 forecasts',
      'if_gate_passes':{'primary':'full','formal_kinds':['lstm','trend','short','fixed_mix','mix_noshrink','full']},
      'if_gate_fails':{'primary':'short_state','formal_kinds':['lstm','trend','short_plain','short_const','short_state'],
          'reason':'No supported two-stream complementarity: test a compact local model and constant/state shrinkage, as a versioned single-stream revision permitted by the plan and required by the active deliverable goal'},
      'hypothesis_screen':'probe change must reduce endpoint MSE in both earliest folds to be a supported screening direction',
      'states':'past-60-return standard deviation high/low; median computed from inner-fit origins in each fold',
      'performance_targets':{'minimum_pooled_improvement_pct':1,'all_seeds_above_zero':True,
          'minimum_positive_fold_means':4,'comparators':['zero','lstm','linear_same_features','lstm_constant_shrink'],
          'paired_block_intervals':{'repetitions':5000,'confidence':.95,'session_block_lengths':[1,5],'seed':2026}},
      'no_independent_confirmation':True,'Q3_values_read':False,'previous_scores_seen':True,
      'changes_not_allowed_after_scores':['new seeds','new horizon selection','dropping adverse folds','undisclosed model search']})
    (BASE/'protocol/IMPLEMENTATION_ADDENDUM.md').write_text('''# 执行前补充登记

用户已通过本轮 active goal 指示执行计划并交付小模型、完整对照和英文论文，包括失败的否定结果。原计划保留双路缺乏依据时另立单路修订的入口。本补充在任何新增市场拟合前固定该条件路径，避免事后选型。

双路互补筛查：xPatch 趋势/残差单路在拟合段定义的高、低波动状态中出现相反排名，并在最早两折保持同样方向。其成立只是开发筛查，不是已经证明融合有效。

若筛查不成立，双路候选不进入正式拟合，主模型固定为局部卷积单路加状态收缩；正式矩阵为原 LSTM、趋势单路、局部无收缩、局部常数收缩、局部状态收缩。共五配置×五折×三种子=75格，原 LSTM 五格核验复用后需70次新正式拟合，加12次探针合计82次。双路成立则沿用原六组90格，预计97次新拟合。原最坏102次上限保持。

单路修订检验的是预测幅度与状态收缩假设，不把失败的双路互补假设包装成成功。全部保留原始探针和正式对照结果。原 LSTM 按原配方重训另外两个种子；候选允许零初始化早停选项，其差异披露，并必须超过零与常数收缩控制才能主张预测增量。

新增时间和进程预算来自本次执行指令；不延长原核心队列、不恢复已终止的另一个融合项目。保护期间仍不启封。
''')
    print(json.dumps({'status':'frozen','probe_jobs':len(jobs),'deadline_epoch':now+86400,'source_hashes':code_hashes()}))

if __name__=='__main__':main()
