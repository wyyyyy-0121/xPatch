"""Measure original CPU acceptance from kernel process birth to durable output.

This external adapter preserves the frozen engine and its original timings.
Linux boot time avoids the WSL wall-clock changes seen by the waiting receiver.
"""
import functools
import math
import os
from pathlib import Path
import runpy
import sys
import time

from fusion.next.common import ROOT, OUT, digest, identity, read_json, source_identity


POLICY_NAME = 'original_delivery_timing_policy.json'
SCOPE = 'Linux process birth through first durable prediction, including interpreter startup, imports, admission, model load, input preparation and warmup'
_installed = False


def process_birth(pid=None):
    pid = os.getpid() if pid is None else pid
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid':pid, 'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'start_ticks':int(stat[19]), 'ticks_per_second':os.sysconf('SC_CLK_TCK')}


def elapsed_from_birth(birth):
    if process_birth() != birth:
        raise ValueError('Cold-start process kernel identity changed')
    observed = time.clock_gettime(time.CLOCK_BOOTTIME)
    elapsed = observed - birth['start_ticks']/birth['ticks_per_second']
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Invalid full-process cold-start interval')
    return elapsed, observed


def specification():
    from scripts.portable_inference_runtime import checked_policy as numerical
    from scripts.future_speed_runtime import checked_policy as speed
    from scripts.portable_package_runtime import checked_policy as package
    value = {'version':'original-delivery-full-process-cold-v1', 'source_identity':source_identity(),
             'numerical_policy_identity':numerical()['identity'], 'speed_policy_identity':speed()['identity'],
             'package_policy_identity':package()['identity'], 'runtime_sha256':digest(Path(__file__)),
             'package_entry_bridge_sha256':digest(ROOT/'scripts/original_cpu_package_entry.py'),
             'package_module_entry':'fusion.next.local_package -> scripts.original_cpu_package_entry',
             'amendment_sha256':digest(ROOT/'fusion/next/ORIGINAL_DELIVERY_TIMING_AMENDMENT.md'),
             'measurement_scope':SCOPE, 'clock':'CLOCK_BOOTTIME minus /proc start_ticks / SC_CLK_TCK',
             'clock_resolution_policy':'kernel birth tick truncation conservatively adds less than one tick',
             'cold_seconds_ceiling':60, 'original_timings_preserved':True,
             'model_or_predictions_changed':False, 'training_changed':False,
             'research_gates_changed':False, 'absolute_deadline_changed':False}
    value['identity'] = identity(value)
    return value


def checked_policy():
    value = read_json(OUT/'performance'/POLICY_NAME)
    if value != specification():
        raise ValueError('Original delivery cold-start policy or implementation differs')
    return value


def validate_measurements(report, progress, policy):
    """Fail closed on missing, inconsistent or legacy function-only timing."""
    rows = report.get('cold_load_measurements_by_process', [])
    if (report.get('cold_timing_policy_identity') != policy['identity']
            or report.get('cold_scope') != policy['measurement_scope']
            or report.get('cold_timing_complete') is not True
            or rows != progress.get('processes') or len(rows) < 2):
        raise ValueError('Full-process cold-start policy or measurement chain differs')
    keys = []
    durations = []
    for row in rows:
        birth = row.get('kernel_process_birth', {})
        elapsed = row.get('full_process_to_first_output_seconds')
        observed = row.get('first_durable_output_boottime_seconds')
        if (not isinstance(birth.get('pid'), int) or birth['pid'] != row.get('pid')
                or not isinstance(birth.get('boot_id'), str) or not birth['boot_id']
                or not isinstance(birth.get('start_ticks'), int) or birth['start_ticks'] <= 0
                or not isinstance(birth.get('ticks_per_second'), int) or birth['ticks_per_second'] <= 0
                or row.get('cold_timing_policy_identity') != policy['identity']
                or not isinstance(elapsed, (float, int)) or isinstance(elapsed, bool)
                or not math.isfinite(elapsed) or elapsed < 0
                or not isinstance(observed, (float, int)) or not math.isfinite(observed)
                or abs(observed - birth['start_ticks']/birth['ticks_per_second'] - elapsed) > 1e-7):
            raise ValueError('Missing or inconsistent kernel cold-start measurement')
        keys.append((birth['boot_id'], birth['pid'], birth['start_ticks']))
        durations.append(elapsed)
    if len(set(keys)) != len(keys) or report.get('distinct_exit_and_resume_processes') is not True:
        raise ValueError('Cold-start exit/resume measurements require distinct processes')
    full = max(durations)
    engine = report.get('engine_cold_load_seconds')
    if (report.get('cold_load_seconds') != full or not isinstance(engine, (float, int))
            or not math.isfinite(engine) or engine < 0 or full + 1e-7 < engine):
        raise ValueError('Full-process cold-start maximum or retained engine timing differs')
    return full


def install():
    global _installed
    if _installed:
        return
    policy = checked_policy()
    from scripts import portable_package_runtime as package
    if not package._installed:
        raise RuntimeError('Install the verified numerical, speed and package adapters first')
    from fusion.next import common, stream, local_package
    birth = process_birth()
    first_outputs = {}
    original_predict = stream.Stream.predict
    original_write = common.write_json

    def acceptance_directory(path):
        path = Path(path).resolve()
        return path.parent if path.parent.parent.name == 'deployment_acceptance' else None

    @functools.wraps(original_predict)
    def predict(self, *args, **kwargs):
        result = original_predict(self, *args, **kwargs)
        directory = acceptance_directory(self.state_path)
        # Stream.predict has already fsynced both the state and directory.
        if directory is not None and result['status'] == 'ok' and directory not in first_outputs:
            if not self.state_path.is_file():
                raise ValueError('First cold-start prediction has no durable state')
            first_outputs[directory] = elapsed_from_birth(birth)
        return result

    @functools.wraps(original_write)
    def write_json(path, value):
        directory = acceptance_directory(path)
        if directory is not None and Path(path).name == 'progress.json' and 'processes' in value:
            for row in value['processes']:
                if row['pid'] == birth['pid']:
                    row.update(kernel_process_birth=birth, cold_timing_policy_identity=policy['identity'])
                    if directory in first_outputs:
                        elapsed, observed = first_outputs[directory]
                        row.update(full_process_to_first_output_seconds=elapsed,
                                   first_durable_output_boottime_seconds=observed)
        if directory is not None and Path(path).name == 'report.json' and 'warm_single_origin_p95_seconds' in value:
            rows = value['cold_load_measurements_by_process']
            value.update(engine_cold_load_seconds=value['cold_load_seconds'], cold_scope=policy['measurement_scope'],
                         cold_timing_policy_identity=policy['identity'], cold_timing_complete=True)
            value['cold_load_seconds'] = max(row.get('full_process_to_first_output_seconds', float('inf')) for row in rows)
            validate_measurements(value, read_json(directory/'progress.json'), policy)
            value['engineering_acceptance_passed'] = bool(value['engineering_acceptance_passed']
                and value['cold_load_seconds'] <= policy['cold_seconds_ceiling'])
        return original_write(path, value)

    stream.Stream.predict = predict
    common.write_json = write_json
    # Avoid importing delivery during sitecustomize: -m must execute its own
    # module with these bindings. Also support already imported test callers.
    delivery = sys.modules.get('fusion.next.delivery')
    if delivery is not None:
        delivery.write_json = write_json
    original_files = local_package.source_files
    def files():
        return {**original_files(), 'evidence/'+POLICY_NAME:OUT/'performance'/POLICY_NAME,
                'evidence/original_delivery_timing_runtime.py':Path(__file__),
                'evidence/original_cpu_package_entry.py':ROOT/'scripts/original_cpu_package_entry.py',
                'evidence/ORIGINAL_DELIVERY_TIMING_AMENDMENT.md':ROOT/'fusion/next/ORIGINAL_DELIVERY_TIMING_AMENDMENT.md'}
    local_package.source_files = files
    # CPython's -m execution otherwise creates a second set of module globals,
    # losing the canonical module's numerical/copy/source-file adapters. Route
    # only the original package CLI through an explicit bridge to that module.
    original_main = runpy._run_module_as_main
    @functools.wraps(original_main)
    def run_main(name, alter_argv=True):
        if name == 'fusion.next.local_package':
            name = 'scripts.original_cpu_package_entry'
        return original_main(name, alter_argv)
    runpy._run_module_as_main = run_main
    _installed = True
