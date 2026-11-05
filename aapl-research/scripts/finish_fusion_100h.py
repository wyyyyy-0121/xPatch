"""Wait for the bound cloud run, then verify and deliver its exact local artifact.

This outer script never changes the admitted model engine or launches training.
The frozen final-test entry point retains exclusive authority over Q3 access.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

import psutil

from fusion.next.common import ROOT, OUT, PLAN, digest, identity, read_json, source_identity, utc_now, write_json

HERE = OUT / 'cloud/local_completion'
REMOTE_ROOT = '/root/autodl-tmp/aapl-fusion'
SSH_OPTIONS = ['-i', '/home/ayaka/.ssh/id_ed25519_qct', '-o', 'IdentitiesOnly=yes',
    '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o',
    'UserKnownHostsFile='+str(ROOT/'tmp/autodl_known_hosts'), '-o', 'ConnectTimeout=10',
    '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3']
HOST = 'root@connect.bjb2.seetacloud.com'
OBSERVER = """import json,pathlib,psutil
root=pathlib.Path('/root/autodl-tmp/aapl-fusion/outputs/fusion_next')
def read(name):
 p=root/name
 return json.loads(p.read_text()) if p.exists() else None
dispatch=read('cloud/bounded_run_dispatch.json')
def alive(receipt):
 try:
  p=psutil.Process(receipt['pid'])
  return abs(p.create_time()-receipt['created_unix'])<.01 and p.is_running() and p.status()!='zombie'
 except psutil.NoSuchProcess:return False
limiter=read('cloud/hard_time_limit_state.json')
print(json.dumps({'dispatch':dispatch,'state':read('cloud/bounded_run_state.json'),
 'limit':read('cloud/hard_time_limit.json'),'limiter':limiter,
 'receipt':read('cloud/return_receipt.json'),'runner_alive':alive(dispatch),'limiter_alive':alive(limiter)}))
"""


def seconds_left(limit):
    return (datetime.fromisoformat(limit['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()


def check_binding(snapshot, binding):
    dispatch = snapshot['dispatch']
    for key in ('pid', 'created_unix', 'scope_identity', 'script_sha256'):
        if dispatch[key] != binding['dispatch'][key]:
            raise ValueError('Cloud execution identity changed: '+key)
    if snapshot['limit'] != binding['limit']:
        raise ValueError('Cloud time limit changed')
    state = snapshot['state']
    if state['scope_identity'] != binding['scope_identity'] or state['source_identity'] != binding['source_identity']:
        raise ValueError('Cloud state belongs to another scope or engine')
    if state['stage'] == 'stopped_for_inspection':
        raise RuntimeError('Cloud runner stopped for inspection: '+state.get('error_type', 'unknown'))
    ready = state['stage'] == 'F6_complete_return_ready'
    if ready:
        receipt = snapshot['receipt']
        if not receipt or receipt['source_identity'] != binding['source_identity']:
            raise ValueError('Completed return receipt is missing or changed')
        if receipt['archive'] != 'outputs/fusion_next/cloud/return_bundle.tar':
            raise ValueError('Unexpected remote return archive')
    elif not snapshot['runner_alive']:
        raise RuntimeError('Bound cloud runner exited without a completed return')
    elif not snapshot['limiter_alive']:
        raise RuntimeError('Bound cloud time limiter is no longer alive')
    return ready


@contextmanager
def return_compatibility():
    """Admit only the exact diagnostic already emitted by the frozen exporter.

    cloud.return_bundle includes batch_scaling.json; restore.FIXED omitted it.
    Preserve the admitted source identity and every original restore safeguard.
    """
    from fusion.next import restore
    original = restore.FIXED
    restore.FIXED = original | {'batch_scaling.json'}
    try:
        yield restore
    finally:
        restore.FIXED = original


def preflight_archive(archive, receipt, binding):
    """Check amended scope and all members before canonical evidence is written."""
    from scripts.report_fusion_scope import inspect
    if digest(archive) != receipt['sha256'] or archive.stat().st_size != receipt['bytes']:
        raise ValueError('Downloaded return archive identity differs')
    with return_compatibility() as restore, tempfile.TemporaryDirectory(dir=ROOT/'tmp', prefix='completion_preflight_') as temporary:
        staging = Path(temporary)
        with tarfile.open(archive) as stream:
            seen = set()
            for member in stream.getmembers():
                if not member.isfile() or member.name in seen or not (member.name == restore.MANIFEST or restore.allowed(member.name)):
                    raise ValueError('Unexpected or duplicate return member: '+member.name)
                if not (staging/member.name).resolve().is_relative_to(staging):
                    raise ValueError('Return member leaves staging')
                seen.add(member.name)
            if restore.MANIFEST not in seen:
                raise ValueError('Return manifest missing')
            stream.extractall(staging, filter='data')
        manifest = read_json(staging/restore.MANIFEST)
        if manifest['source_identity'] != binding['source_identity'] or manifest['plan_sha256'] != binding['plan_sha256']:
            raise ValueError('Return engine or plan differs')
        if any(manifest[k] for k in ('Q3_data_included', 'credentials_included', 'raw_development_market_data_included')):
            raise ValueError('Return declares excluded data')
        if seen != {restore.MANIFEST, *manifest['files']}:
            raise ValueError('Return manifest member set differs')
        for name, row in manifest['files'].items():
            path = staging/name
            if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
                raise ValueError('Return member identity differs: '+name)
        report = inspect(staging/'outputs/fusion_next')
        if report['scope_identity'] != binding['scope_identity'] or not all(r['completed'] for r in report['stages'].values()):
            raise ValueError('Return does not complete the bound reduced scope')
        write_json(HERE/'archive_preflight.json', report)


class Completion:
    def __init__(self, binding):
        self.binding = binding

    def state(self, stage, **details):
        write_json(HERE/'state.json', {'stage':stage, 'updated_at_utc':utc_now(),
            'scope_identity':self.binding['scope_identity'], 'source_identity':self.binding['source_identity'], **details})

    def command(self, command, label, **kwargs):
        remaining = seconds_left(self.binding['limit'])
        if remaining <= 0:
            raise TimeoutError('The original 100-hour deadline has elapsed')
        log = HERE/'logs'/(label+'.log');log.parent.mkdir(parents=True, exist_ok=True)
        managed = os.environ.get('AAPL_COMPLETION_MANAGED_GROUP') == '1'
        with log.open('ab') as output:
            process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=not managed, **kwargs)
            try:
                returncode = process.wait(timeout=remaining)
            except BaseException:
                # Include delivery's replay children; a parent-only timeout can
                # otherwise leave local inference alive beyond the wall cap.
                try:
                    if managed:process.terminate()
                    else:os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        if managed:process.kill()
                        else:os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.wait()
                raise
            finally:
                if not managed:
                    # Clean up descendants even when a worker exits before its
                    # children. Nested finish-local commands share this group.
                    try:os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:pass
        if returncode:
            raise RuntimeError('Completion command failed; inspect '+str(log.relative_to(ROOT)))

    def fetch(self, receipt):
        if shutil.disk_usage(ROOT).free < 3*receipt['bytes']+2**30:
            raise RuntimeError('Insufficient local disk for archive, staging and restored evidence')
        archive = HERE/'return_bundle.tar'
        if not archive.exists() or digest(archive) != receipt['sha256']:
            partial = archive.with_suffix('.tar.partial')
            self.command(['scp', '-P', '19046', *SSH_OPTIONS,
                HOST+':'+REMOTE_ROOT+'/'+receipt['archive'], str(partial)], 'download')
            if digest(partial) != receipt['sha256'] or partial.stat().st_size != receipt['bytes']:
                raise ValueError('Downloaded archive differs; partial file preserved')
            os.replace(partial, archive)
        write_json(HERE/'return_receipt.json', receipt)
        return archive

    def local_steps(self, archive, receipt):
        preflight_archive(archive, receipt, self.binding)
        self.state('restoring_verified_return')
        with return_compatibility() as restore:
            result = restore.restore(archive, receipt['sha256'])
        write_json(HERE/'restore_receipt.json', result)
        from scripts.report_fusion_scope import run as scope_report
        scope_report()
        # The original entry rechecks raw development and Q2 predictions before
        # any Q3 acquisition. A negative path proceeds directly to engineering.
        from fusion.next.confirmation import checked_freeze
        from fusion.next.freeze import recheck_seed_predictions, recheck_controls
        from fusion.next.panel import Panel
        _, frozen = checked_freeze()
        selected, _, _, _ = recheck_seed_predictions(Panel.load(), frozen['selected_system'])
        controls, simple = recheck_controls(Panel.load())
        checks = {
            'pooled_improvement_at_least_1pct': selected['system']['mse_improvement_vs_zero'] >= .01,
            'beats_fixed_strongest_single': selected['system']['mse'] < controls[0]['mse'],
            'beats_fixed_strongest_simple': selected['system']['mse'] < simple['mse'],
            'at_least_two_validation_quarters_improve': sum(r['mse_improvement_vs_zero'] > 0 for r in selected['quarter_scores'].values()) >= 2,
            'all_three_seeds_improve': all(r['mse_improvement_vs_zero'] > 0 for r in selected['individual_seeds'].values())}
        if checks != frozen['development_gate'] or all(checks.values()) != frozen['development_gate_passed']:
            raise ValueError('Local development gate differs from the frozen decision')
        if controls[0] != frozen['fixed_strongest_single'] or simple != frozen['fixed_strongest_simple']:
            raise ValueError('Recomputed strongest controls differ')
        confirmation = read_json(OUT/'confirmation.json')
        if confirmation['system_freeze_sha256'] != digest(OUT/'system_freeze.json'):
            raise ValueError('Confirmation belongs to another system')
        proceed = choose_final(frozen, confirmation)
        deployment = OUT/frozen['final_fit_policy']['artifact']
        if deployment.resolve() != (OUT/'deployment/pre_Q2').resolve():
            raise ValueError('Unexpected final deployment path')
        prefix = [sys.executable, '-m']
        if proceed:
            if not (OUT/'final_test/report.json').exists():
                self.state('conditional_F7_running')
                self.command(prefix+['fusion.next.final_test', '--deployment', str(deployment)], 'final_test')
            self.command(prefix+['fusion.next.research_report', '--recompute-final'], 'final_statistics')
        elif (OUT/'final_test/entry.json').exists():
            raise ValueError('Final entry exists on a negative path')
        self.state('F8_local_replay', final_test_requested=proceed)
        manifest = read_json(deployment/'deployment.json')
        acceptance = OUT/'deployment_acceptance'/manifest['identity']/'report.json'
        if not acceptance.exists():
            self.command(prefix+['fusion.next.delivery', '--deployment', str(deployment), '--origins', '256'], 'delivery')
        self.state('F8_package_verification')
        package = OUT/'local_packages'/(manifest['identity']+'.tar')
        if package.exists():
            from fusion.next.local_package import verify
            verify(package)
        else:
            self.command(prefix+['fusion.next.local_package', '--deployment', str(deployment), '--verify'], 'package')
        self.command(prefix+['fusion.next.research_report'], 'research_report')
        report = read_json(OUT/'research_report.json')
        if report['violations']:
            raise ValueError('Research report detected a protocol violation')
        self.state('delivery_evidence_ready', research_status=report['research_status'],
            engineering_acceptance_passed=read_json(acceptance)['engineering_acceptance_passed'],
            Q3_preserved_on_negative_path=report['Q3_preserved_on_negative_path'],
            user_review_still_required=True, rental_shutdown_performed=False)
        self.publish_summary(report, manifest, package)

    def publish_summary(self, report, manifest, package):
        # Publish only a small result summary; raw prices, fitted weights and
        # machine-specific receipts stay in ignored research output folders.
        path = ROOT/'fusion/next/BOUNDED_DELIVERY_RESULT.md'
        acceptance = report['engineering_acceptance']
        lines = ['# 100 小时核心融合执行结果', '',
            '范围：两轮、三个验证季度、三个种子；压缩搜索及部分机制对照。', '',
            f'- 范围身份：`{self.binding["scope_identity"]}`。',
            f'- 系统身份：`{manifest["identity"]}`。',
            f'- 研究状态：`{report["research_status"]}`。',
            f'- 开发门槛通过：`{read_json(OUT/"system_freeze.json")["development_gate_passed"]}`。',
            f'- Q2 确认通过：`{report["Q2_confirmation"]["proceed_to_final_test"]}`。',
            f'- 最终检验完成：`{report["stages"]["F7_evaluated"]}`。',
            f'- CPU 工程验收通过：`{acceptance["engineering_acceptance_passed"]}`。',
            f'- CPU 推理包 SHA256：`{digest(package)}`。', '',
            '预测指标、机制限制、实际运行时间与清理记录见本地 outputs/fusion_next/research_report.json，',
            '精简范围见同目录 bounded_scope_report.json；原始价格、模型权重与连接信息不上传 GitHub。', '',
            '完成推理包校验不代表可交易性或期刊录用。平台租赁实例未由本脚本关机。', '']
        content = '\n'.join(lines)
        if path.exists() and path.read_text() != content:
            raise ValueError('Existing delivery summary differs; preserve it for inspection')
        branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
        remote = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=ROOT, text=True).strip()
        if branch != 'codex/aapl-return12-implementation' or remote != 'git@github.com:wyyyyy-0121/xPatch.git':
            raise ValueError('Git destination changed; delivery summary not published')
        path.write_text(content, encoding='utf-8')
        self.command(['git', 'add', '--', str(path)], 'git_add_summary')
        changed = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', str(path)], cwd=ROOT).returncode
        if changed == 1:
            self.command(['git', '-c', 'user.name=wyyyyy-0121', '-c', 'user.email=hi.universe0121@gmail.com',
                'commit', '--only', '-m', 'Record bounded fusion research and CPU delivery outcome', '--', str(path)], 'git_commit_summary')
        elif changed != 0:
            raise RuntimeError('Could not inspect summary changes')
        self.command(['git', 'push', 'origin', branch], 'git_push_summary')
        self.state('delivery_evidence_ready_and_summary_pushed', research_status=report['research_status'],
            engineering_acceptance_passed=acceptance['engineering_acceptance_passed'],
            Q3_preserved_on_negative_path=report['Q3_preserved_on_negative_path'],
            user_review_still_required=True, rental_shutdown_performed=False)


def choose_final(frozen, confirmation):
    if confirmation.get('Q3_opened'):
        raise ValueError('Cloud confirmation declares Q3 access')
    if not frozen['development_gate_passed']:
        if confirmation.get('evaluated') or confirmation.get('proceed_to_final_test'):
            raise ValueError('Confirmation contradicts the negative development gate')
        return False
    if not confirmation.get('evaluated'):
        raise ValueError('Passing development gate has no completed Q2 evaluation')
    if not isinstance(confirmation.get('proceed_to_final_test'), bool):
        raise ValueError('Missing Q2 gate decision')
    return confirmation['proceed_to_final_test']


def observe():
    result = subprocess.run(['ssh', '-p', '19046', *SSH_OPTIONS, HOST,
        REMOTE_ROOT+'/.venv/bin/python -'], input=OBSERVER, text=True,
        capture_output=True, timeout=45)
    if result.returncode:
        raise ConnectionError('Cloud observation failed (SSH exit '+str(result.returncode)+')')
    return json.loads(result.stdout)


def run():
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE/'execution.lock').open('a') as lock:
        try:fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Local completion observer is already active') from None
        snapshot = observe()
        path = HERE/'binding.json'
        if path.exists():binding = read_json(path)
        else:
            if snapshot['limit']['total_wall_hours'] != 100:
                raise ValueError('Expected the authorized 100-hour limit')
            saved = read_json(OUT/'cloud/replan_100h/bounded_run_dispatch.json')
            scope = read_json(OUT/'cloud/replan_100h/scope_100h.json')
            if scope['identity'] != identity({k:v for k,v in scope.items() if k != 'identity'}):
                raise ValueError('Local approved scope identity differs')
            write_json(HERE/'hard_time_limit.json', snapshot['limit'])
            if digest(HERE/'hard_time_limit.json') != scope['hard_limit_receipt_sha256']:
                raise ValueError('Remote limit differs from the approved scope')
            binding = {'dispatch':saved, 'scope_identity':scope['identity'],
                'source_identity':source_identity(), 'plan_sha256':digest(PLAN),
                'limit':snapshot['limit'], 'script_sha256':digest(Path(__file__)),
                'created_at_utc':utc_now()}
            check_binding(snapshot, binding)
            write_json(path, binding)
        if binding['script_sha256'] != digest(Path(__file__)) or binding['source_identity'] != source_identity() or binding['plan_sha256'] != digest(PLAN):
            raise ValueError('Local completion code, engine or plan changed')
        completion = Completion(binding)
        write_json(HERE/'dispatch.json', {'pid':os.getpid(), 'created_at_utc':utc_now(),
            'created_unix':psutil.Process().create_time(),
            'script_sha256':binding['script_sha256'], 'scope_identity':binding['scope_identity']})
        failures = 0
        try:
            while True:
                if seconds_left(binding['limit']) <= 0:
                    raise TimeoutError('Original hard deadline reached; no further work dispatched')
                ready = check_binding(snapshot, binding)
                write_json(HERE/'last_cloud_snapshot.json', snapshot)
                if ready:
                    completion.state('downloading_completed_return')
                    archive = completion.fetch(snapshot['receipt'])
                    completion.command([sys.executable, str(Path(__file__).resolve()), '--finish-local'], 'local_steps',
                        env={**os.environ, 'AAPL_COMPLETION_MANAGED_GROUP':'1'})
                    return
                completion.state('waiting_for_cloud', cloud_stage=snapshot['state']['stage'])
                time.sleep(min(60, max(0, seconds_left(binding['limit']))))
                try:
                    snapshot = observe();failures = 0
                except (ConnectionError, subprocess.TimeoutExpired):
                    failures += 1
                    completion.state('cloud_connection_retry', consecutive_failures=failures)
                    if failures >= 10:raise
        except BaseException as error:
            completion.state('stopped_for_inspection', error_type=type(error).__name__, error=str(error))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finish-local', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.finish_local:
        binding = read_json(HERE/'binding.json')
        if binding['script_sha256'] != digest(Path(__file__)) or binding['source_identity'] != source_identity() or binding['plan_sha256'] != digest(PLAN):
            raise ValueError('Local completion code, engine or plan changed')
        if seconds_left(binding['limit']) <= 0:raise TimeoutError('Original deadline reached')
        Completion(binding).local_steps(HERE/'return_bundle.tar', read_json(HERE/'return_receipt.json'))
    else:run()
