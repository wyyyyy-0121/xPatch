"""Freeze measured input acceleration independently of the original engine."""
from fusion.next.common import OUT, ROOT, digest, identity, read_json, source_identity, utc_now, write_json
from scripts.portable_inference_runtime import checked_policy as portable_policy


def prepare():
    runtime = ROOT/'scripts/future_speed_runtime.py'
    launcher = ROOT/'scripts/start_fast_fusion.py'
    reports = {}
    for location in ('local', 'cloud'):
        path = OUT/'cloud/speed_preparation'/(location+'_benchmark.json')
        report = read_json(path)
        if (report['base_engine_source_identity'] != source_identity()
                or report['runtime_sha256'] != digest(runtime)
                or not report['exact_input_parity_passed'] or report['warm_speedup'] <= 1):
            raise ValueError('Input acceleration benchmark is missing or stale')
        reports[location] = (path, report)
    policy = {'version': 'future-pointwise-background-v1', 'at_utc': utc_now(),
              'base_engine_source_identity': source_identity(),
              'numerical_policy_identity': portable_policy()['identity'],
              'runtime_sha256': digest(runtime), 'launcher_sha256': digest(launcher),
              'local_benchmark_sha256': digest(reports['local'][0]),
              'cloud_benchmark_sha256': digest(reports['cloud'][0]),
              'exact_input_parity_passed': True,
              'warm_speedup': min(value[1]['warm_speedup'] for value in reports.values()),
              'training_recipe_changed': False, 'sampler_or_rng_changed': False,
              'inference_precision_changed': False, 'Q3_accessed': False,
              'hard_deadline_changed': False,
              'authorization': 'User: 开始优化 把优化好的东西往后接入就好了',
              'activation': 'explicit future entry and future local completion children only',
              'current_cloud_process_hot_patched': False,
              'measurement_limit': 'CPU input preparation only, not end-to-end training speed'}
    path = OUT/'performance/future_speed_policy.json'
    if path.exists():
        saved = read_json(path)
        if any(saved.get(k) != v for k, v in policy.items() if k != 'at_utc'):
            raise ValueError('Existing speed policy differs; preserve it')
        return saved
    policy['identity'] = identity(policy)
    for location, (source, _) in reports.items():
        (OUT/'performance'/('future_speed_'+location+'_benchmark.json')).write_bytes(source.read_bytes())
    for source in (runtime, launcher):
        (OUT/'performance'/(source.name+'.txt')).write_bytes(source.read_bytes())
    write_json(path, policy)
    return policy


if __name__ == '__main__':
    print(prepare())
