"""Recompute optional planning estimates from the frozen evidence, without training."""
import math
from fusion.next.common import OUT,read_json,write_json,digest,source_identity,utc_now

spec=read_json(OUT/'search_manifest.json');gpu=read_json(OUT/'performance_admission.json');cpu=read_json(OUT/'performance_cpu.json')
if not gpu.get('passed') or spec['source_identity']!=source_identity():raise RuntimeError('Final F2/search source is not admitted')
policy=spec['training_budget_policy']
if policy['total_coverage_equivalent_epochs']!=2:raise RuntimeError('The user requires two total equivalent epochs')
projection=spec['projection'];m=gpu['measurements'];precision=spec['groups']['fusion'][0]['precision']
suffix='_fp32' if precision=='fp32' else ''
def timing(name):return m.get(name+suffix,m[name])
cold=max(timing(k)['seconds_per_update'] for k in ('end_to_end_raw','end_to_end_cache_population','frozen_tcn_end_to_end'))
warm=timing('end_to_end_cache_reuse')['seconds_per_update']
# TCN cached training was not separately measured. This is an explicit scenario,
# conditioned on both prescribed heads sustaining the measured reuse rate.
cache_maps=6
examples=policy['largest_fit_data']['eligible_unique_windows'];batch=spec['groups']['fusion'][0]['batch_size']
fill_seconds=cache_maps*math.ceil(examples/batch)*timing('end_to_end_cache_population')['seconds_per_update']
saved_seconds=sum(row['updates']*max(0,cold-warm) for row in projection['work'] if row['family']=='fusion')
margin=projection['margin_multiplier'];base=projection['formal_wall_hours_projection_with_margin']
conditional=base-margin*saved_seconds/3600+margin*fill_seconds/3600
profile=read_json(OUT/'execution_profile.json');price=profile['price_cny_per_hour_quoted']
value={'created_at_utc':utc_now(),'source_identity':source_identity(),'search_identity':spec['identity'],
 'search_manifest_sha256':digest(OUT/'search_manifest.json'),'measurement_gpu_sha256':digest(OUT/'performance_admission.json'),
 'measurement_cpu_sha256':digest(OUT/'performance_cpu.json'),'training_budget_policy':policy,
 'counts':spec['counts'],'batch_sizes':policy['batch_sizes'],'precision_by_family':{k:spec['groups'][k][0]['precision'] for k in ('xpatch','patchtst','fusion','lora')},
 'price_cny_per_hour_quoted':price,'margin_multiplier':margin,
 'conservative':{'hours':base,'days':base/24,'cost_at_quote_cny':base*price,'category_hours':projection['category_hours_with_margin']},
 'conditional_sustained_cache_reuse':{'hours':conditional,'days':conditional/24,'cost_at_quote_cny':conditional*price,
  'cache_maps':cache_maps,'first_population_hours_before_margin':fill_seconds/3600,
  'assumptions':['same source, fixed backbone weights and head precision throughout',
   'all three lookbacks times two information scopes fit in persistent cache storage',
   'both frozen market encoders sustain the measured Transformer cached rate; TCN cached rate has not been measured separately',
   'random backbone and LoRA use raw execution; no discount for removed modules or CPU/GPU overlap'],
  'not_a_verified_lower_bound_or_completion_promise':True},
 'actual_provider_invoice_available':False,'Q3_accessed':False}
write_json(OUT/'cloud/cost_two_epoch_final.json',value)
print({k:value[k] for k in ('batch_sizes','precision_by_family','conservative','conditional_sustained_cache_reuse')})
