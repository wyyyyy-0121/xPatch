"""User-authorized scope amendment using the unchanged, admitted training engine."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import fcntl
from pathlib import Path

from fusion.next.common import OUT,digest,identity,read_json,source_identity,utc_now,write_json
from fusion.next.design import halving_units


SCOPE=OUT/'performance/scope_100h.json'


def plan(original):
    """Select fixed list positions and adapter coordinates, never observed scores."""
    if original['training_budget_policy']['total_coverage_equivalent_epochs']!=2:
        raise ValueError('The user retains two total equivalent epochs')
    spec=copy.deepcopy(original)
    for family,count in (('lightgbm',4),('xpatch',4),('patchtst',4),('fusion',2)):
        spec['groups'][family]=spec['groups'][family][:count]
    spec['groups']['lora']=[cfg for cfg in spec['groups']['lora']
        if cfg['promoted_structure_slot']==0 and cfg['backbone_lr']==1e-5]
    if len(spec['groups']['lora'])!=2 or {cfg['rank'] for cfg in spec['groups']['lora']}!={8,16}:
        raise ValueError('Need two fixed ranks of the same promoted structure and adapter rate')
    spec['transfer_configs']=[]
    work=[]
    for old in original['projection']['work']:
        row=copy.deepcopy(old);description=old['description'];family=old['family'];factor=0.
        if description=='same-information full CPU search':
            factor=1. if family=='ridge' else 4/len(original['groups'][family])
        elif description=='supervised search 10/30/100% prefixes':
            factor=halving_units(4)/halving_units(len(original['groups'][family]))
        elif description=='quarterly OOF main OHLCV':factor=1.
        elif description=='neural search':
            factor=halving_units(len(spec['groups'][family]))/halving_units(len(original['groups'][family]))
        elif description=='top two, three seeds':
            # Only one selected structure gets seed review; its seed 2026 final
            # prefix is already in the identical search namespace and recipe.
            factor=1/3
            row['description']='one selected structure, two additional seeds; seed 2026 reused'
        elif description=='mechanism plain_mlp (no speed discount for removed modules)':factor=.5
        elif old['category']=='final_and_deployment':
            # Original projection reserved two full final fits. The actual
            # policy has exactly one pre-Q2 system fit, then no Q2 refit.
            factor=.5
        if not factor:continue
        for key in ('fit_equivalents','updates','predict_rows','projected_seconds_before_margin'):
            row[key]*=factor
        row['original_work_fraction']=factor;work.append(row)
    margin=original['projection']['margin_multiplier']
    categories={}
    for row in work:
        category=row['category']
        categories[category]=categories.get(category,0)+margin*row['projected_seconds_before_margin']/3600
    reserve=4. # recovery plus local replay/package and queue overhead; no overlap credit
    hours=sum(categories.values())+reserve
    return {'parent_search_identity':original['identity'],'counts':{family:len(configs) for family,configs in spec['groups'].items()},
        'simple_count':len(spec['simple']),'groups':spec['groups'],'simple':spec['simple'],
        'validation':original['validation'],'seeds':original['seeds'],
        'training_budget_policy':original['training_budget_policy'],
        'seed_review_structures_per_family':1,'main_OOF_quarters':13,
        'mechanisms_retained':['gate_vs_plain_mlp','frozen_vs_lora'],
        'mechanisms_deferred':['no_foundation','random_frozen','no_market','close_only','AAPL_vs_multistock'],
        'transfer_pairs':0,'scope_selection':'first fixed list positions; LoRA slot 0, ranks 8/16, rate 1e-5',
        'scores_used_for_scope_selection':False,
        'projection':{'hours_with_margin_and_reserve':hours,'margin_multiplier':margin,
            'category_hours_with_margin':categories,'recovery_and_local_delivery_reserve_hours':reserve,
            'CPU_GPU_overlap_credit':0,'warm_cache_speed_credit':0,'work':work},
        'original_complete_scope_claimed':False,'Q3_accessed':False}


def remaining_hours():
    limit=read_json(OUT/'cloud/hard_time_limit.json')
    if limit['total_wall_hours']!=100:raise ValueError('Missing user-approved 100-hour wall cap')
    return (datetime.fromisoformat(limit['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()/3600


def prepare_scope(original):
    payload=plan(original);script_sha=digest(Path(__file__))
    if SCOPE.exists():
        saved=read_json(SCOPE)
        if saved['identity']!=identity({k:v for k,v in saved.items() if k!='identity'}):raise ValueError('Scope receipt changed')
        if saved['execution_script_sha256']!=script_sha or saved['parent_search_identity']!=original['identity']:
            raise ValueError('Cannot silently change a running bounded experiment')
        return saved
    remaining=remaining_hours()
    if payload['projection']['hours_with_margin_and_reserve']>remaining:
        raise RuntimeError('Measured scope projection no longer fits the remaining hard limit')
    payload.update(source_identity=source_identity(),created_at_utc=utc_now(),
        authorization='User: 保留两轮，压缩搜索与机制（建议）',execution_script_sha256=script_sha,
        parent_search_sha256=digest(OUT/'search_manifest.json'),remaining_hours_at_freeze=remaining,
        hard_limit_receipt_sha256=digest(OUT/'cloud/hard_time_limit.json'))
    payload['identity']=identity(payload);write_json(SCOPE,payload)
    # The existing verified return allowlist includes performance/*.json/*.txt.
    # Preserve the external runner beside the amendment for local reproduction.
    (OUT/'performance/scope_100h_runner.txt').write_bytes(Path(__file__).read_bytes())
    write_json(OUT/'performance/scope_100h_time_limit.json',read_json(OUT/'cloud/hard_time_limit.json'))
    return payload


def run(plan_only=False):
    from fusion.next.experiment import (admitted,search,build_oof,expert_configurations,simple_controls,
        verify_seeds,lora_configs,mechanism_configs)
    from fusion.next.panel import Panel
    from fusion.next.trainer import ExpertStore
    from fusion.next.accounting import recover_interrupted,totals
    from fusion.next.hardware import runtime
    if runtime(2026)!='cuda':raise RuntimeError('Use the admitted cloud runtime')
    lock=(OUT/'cloud/bounded_run.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('A bounded execution is already active')
    original=admitted();scope=prepare_scope(original)
    if plan_only:print(scope['projection']);return
    for family in ('ridge','lightgbm','xpatch','patchtst','fusion'):
        path=OUT/'selections'/('fusion.json' if family=='fusion' else 'single_'+family+'.json')
        if path.exists():
            selection=read_json(path)
            observed={r['config']['candidate_id'] for r in selection['rungs'][0]['rankings']}
            expected={cfg['candidate_id'] for cfg in scope['groups'][family]}
            if observed!=expected:raise ValueError('Saved selection belongs to a different scope: '+family)
    for name in ('F3_complete.json','F4_complete.json','F5_complete.json'):
        if (OUT/name).exists() and read_json(OUT/name).get('scope_identity')!=scope['identity']:
            raise ValueError('Existing stage completion belongs to a different scope: '+name)
    recover_interrupted()
    if totals()['active_gpu_jobs']:raise RuntimeError('Previous GPU job is still alive; no duplicate launch')
    panel=Panel.load()
    spec={**original,'groups':scope['groups'],'simple':scope['simple']}

    def state(stage,**details):
        write_json(OUT/'cloud/bounded_run_state.json',{'stage':stage,'scope_identity':scope['identity'],
            'source_identity':source_identity(),'updated_at_utc':utc_now(),'remaining_hours':remaining_hours(),
            'Q3_accessed':False,**details})

    def complete(name,**details):
        write_json(OUT/name,{'source_identity':source_identity(),'scope_identity':scope['identity'],
            'scope_amendment_sha256':digest(SCOPE),'original_complete_scope_claimed':False,
            'at_utc':utc_now(),**details})

    try:
        if not (OUT/'F3_complete.json').exists():
            state('F3_running')
            def cpu_search():
                for family in ('ridge','lightgbm'):
                    search(panel,spec['groups'][family],'single_'+family,category='experts_and_cache')
            with ThreadPoolExecutor(max_workers=1) as pool:
                future=pool.submit(cpu_search)
                for family in ('xpatch','patchtst'):
                    search(panel,spec['groups'][family],'single_'+family,category='experts_and_cache')
                future.result()
            store=build_oof(panel,expert_configurations(spec));simple_controls(panel,spec,store)
            complete('F3_complete.json',main_OOF=store.key)
        else:store=ExpertStore(panel)
        if not (OUT/'F4_complete.json').exists():
            state('F4_running');search(panel,spec['groups']['fusion'],'fusion',store)
            complete('F4_complete.json',transfer_search='deferred by user-approved 100-hour scope amendment')
        if not (OUT/'F5_complete.json').exists():
            state('F5_running')
            top=read_json(OUT/'selections/fusion.json')['top_two']
            lora=lora_configs(top,spec)
            adapted=search(panel,lora,'lora',store,'adaptation_and_mechanisms')
            frozen_seed=verify_seeds(panel,top[:1],'fusion',store)
            lora_seed=verify_seeds(panel,adapted['top_two'][:1],'lora',store)
            variants=mechanism_configs(top[:1],'plain_mlp')
            plain=verify_seeds(panel,variants,'mechanism_plain_mlp',store)
            reference=copy.deepcopy(frozen_seed[0]);reference['config']['paired_base_id']=top[0]['candidate_id']
            reference['reference_reused_without_retraining']=True
            candidates=frozen_seed+lora_seed+plain
            candidates.sort(key=lambda row:(row['system']['mse'],row['config']['candidate_id']))
            complete('F5_complete.json',candidates=candidates,selected=candidates[0],
                mechanisms={'full':[reference],'plain_mlp':plain},usage=totals(),
                mechanisms_deferred=scope['mechanisms_deferred'],
                lora_interpretation='two fixed ranks of the best frozen structure, one adapter learning rate')
        # Exactly the same frozen-system gate and negative-result export policy.
        if not (OUT/'system_freeze.json').exists():
            state('system_freeze');from fusion.next.freeze import prepare;prepare()
        if not (OUT/'confirmation.json').exists():
            state('F6_running');from fusion.next.confirmation import run as confirm;confirm()
        if not (OUT/'cloud/return_receipt.json').exists():
            state('return_bundle');from fusion.next.cloud import return_bundle;return_bundle()
        state('F6_complete_return_ready',next_step='verify locally, conditional F7, then actual F8 delivery')
    except BaseException as error:
        state('stopped_for_inspection',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args();run(args.plan_only)
