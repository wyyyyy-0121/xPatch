"""Verify the one-time timers and write an honest night-mode receipt."""
import base64
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys
from pathlib import Path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    directory=Path(sys.argv[1]).resolve();snapshot_path=directory/'CHECKPOINT_AND_BUDGET_SNAPSHOT.json'
    snapshot=json.loads(snapshot_path.read_text());plan=json.loads((directory/'RESUME_PLAN.json').read_text())
    resume_time=datetime.fromisoformat(snapshot['resume_at_local'])
    expected_systemd_time=resume_time.strftime('%Y-%m-%d %H:%M:%S')
    expected_windows_time=resume_time.strftime('%Y-%m-%dT%H:%M:%S')
    display_time=resume_time.strftime('%Y-%m-%d %H:%M')
    display_clock=resume_time.strftime('%H:%M')
    validation=json.loads((directory/'CHECKPOINT_RESUME_VALIDATION.json').read_text())
    paused=json.loads((directory/'NIGHT_PAUSE_STATE_RECEIPT.json').read_text())
    if (snapshot['status']!='snapshot_complete' or validation['status']!='all_checkpoint_resume_identities_verified' or
            paused['status']!='queue_statuses_marked_paused' or digest(snapshot_path)!=plan['snapshot_sha256']):
        raise ValueError('Pause or recovery validation is incomplete')
    systemd=subprocess.run(['systemctl','--user','show',plan['systemd_unit']+'.timer',
                            '--property=ActiveState,NextElapseUSecRealtime,AccuracyUSec'],
                           capture_output=True,text=True,timeout=10,check=True)
    props=dict(line.split('=',1) for line in systemd.stdout.splitlines() if '=' in line)
    if props.get('ActiveState')!='active' or expected_systemd_time not in props.get('NextElapseUSecRealtime',''):
        raise ValueError('The systemd timer is not scheduled for the requested time')
    ps=r'''$ErrorActionPreference='Stop'
$task=Get-ScheduledTask -TaskName '__TASK_NAME__'
$info=Get-ScheduledTaskInfo -TaskName '__TASK_NAME__'
[pscustomobject]@{TaskName=$task.TaskName;State=[string]$task.State;StartBoundary=$task.Triggers[0].StartBoundary;NextRunTime=$info.NextRunTime.ToString('o');Executable=$task.Actions[0].Execute;Arguments=$task.Actions[0].Arguments;StartWhenAvailable=$task.Settings.StartWhenAvailable;WakeToRun=$task.Settings.WakeToRun;LogonType=[string]$task.Principal.LogonType} | ConvertTo-Json -Compress'''
    ps=ps.replace('__TASK_NAME__',plan['windows_task'])
    encoded=base64.b64encode(ps.encode('utf-16le')).decode('ascii')
    windows=subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile',
                            '-NonInteractive','-EncodedCommand',encoded],cwd='/mnt/c/Users/hiuni',
                           capture_output=True,timeout=30,check=True)
    lines=windows.stdout.decode('utf8','replace').splitlines()
    record=next((json.loads(line) for line in lines if line.startswith('{')),None)
    if (not record or record['TaskName']!=plan['windows_task'] or record['State']!='Ready' or
            not record['NextRunTime'].startswith(expected_windows_time) or not record['StartWhenAvailable'] or
            record['WakeToRun'] or record['LogonType']!='Interactive' or
            str(snapshot_path) not in record['Arguments'] or str(directory/'RESUME_PLAN.json') not in record['Arguments']):
        raise ValueError('The Windows backup task differs from the expected one-time action')
    manifest=json.loads(Path(snapshot['pause_manifest']).read_text());stopped=[]
    for item in manifest['processes']:
        pid=item['pid'];raw=Path(f'/proc/{pid}/stat').read_text();parts=raw[raw.rfind(')')+2:].split()
        if parts[0] not in ('T','t') or int(parts[19])!=item['start_ticks'] or not os.sched_getaffinity(pid).issubset({24,26}):
            raise ValueError('A paused process is not stopped or exceeds two shared CPUs')
        stopped.append({'pid':pid,'state':parts[0],'affinity':sorted(os.sched_getaffinity(pid))})
    out={'status':'resume_scheduled_not_yet_executed','created_at_utc':datetime.now(timezone.utc).isoformat(),
         'resume_at_local':snapshot['resume_at_local'],'systemd_timer':props,'windows_backup_task':record,
         'paused_processes':stopped,'training_workers':len(snapshot['workers']),
         'worker_devices':[w['device'] for w in snapshot['workers']],
         'checkpoint_snapshot_sha256':digest(snapshot_path),'recovery_plan_sha256':digest(directory/'RESUME_PLAN.json'),
         'queues':{k:{'script':v['script'],'remaining_active_seconds':v['remaining_active_seconds']}
                   for k,v in snapshot['queues'].items()},
         'verification':['all six checkpoint identities checked','timer trigger tested with synthetic process',
                         'budget freeze tested with synthetic process','duplicate trigger lock tested',
                         'Windows-to-WSL dry run passed',f'actual {display_clock} restoration has not yet occurred'],
         'limitations':['A sleeping or powered-off computer is not awakened by these timers.',
                        'The Windows backup uses interactive user login; timing is not guaranteed if logged out.',
                        'If WSL or the operating system shuts down, only the latest periodic on-disk checkpoint survives; later batches may need rerunning.']}
    receipt=directory/'NIGHT_SETUP_RECEIPT.json'
    with receipt.open('x',encoding='utf8') as f:json.dump(out,f,indent=2,ensure_ascii=False);f.write('\n')
    jobs={'4':'H60 收益第 4 折','5':'H60 收益第 5 折','h30_price_f1':'H30 价格第 1 折',
          'h60_price_f1':'H60 价格第 1 折','h5_return_f2':'H5 收益第 2 折',
          'h5_future_close_f2':'H5 价格第 2 折'}
    lines=['# 今晚 AAPL 低负载和自动恢复状态','',
           '当前六个训练任务和四个队列进程均已暂停，所有相关线程限制在同一组两个逻辑 CPU（24、26）；暂停期间训练 CPU 占用为零。',
           '暂停的六个任务：'+ '、'.join(jobs[w['job']] for w in snapshot['workers'])+'。',
           '本次被暂停的训练任务均记录为 CPU 运行；没有启动 AAPL GPU 训练，也未更改其他 GPU 程序。','',
           f'北京时间 {display_time} 的系统定时器和 Windows 一次性备用任务均已注册；重复触发会由锁合并。恢复时优先继续仍冻结的原进程，若 WSL 已关闭则使用最后一个已验证磁盘检查点。','',
           f'暂停时间不计入原队列预算；恢复后按各自剩余预算运行，到期冻结并保留检查点，不自动重置。实际恢复只能在 {display_clock} 后确认。','',
           '电脑需保持开机、唤醒且 Windows 用户保持登录，才能保证这两种触发路径具备运行条件。休眠或关机时不能保证准时恢复；Windows 任务已设置错过后尽可能补运行。若 WSL 或系统提前关闭，最近检查点之后尚未写盘的批次可能需要重做。','',
           '[计划和校验回执]('+str(receipt)+')  ',
           '[检查点与剩余预算]('+str(snapshot_path)+')  ',
           '[计时说明]('+str(directory/'NIGHT_TIMING_NOTE_ZH.md')+')','']
    report=directory/'NIGHT_STATUS_ZH.md'
    with report.open('x',encoding='utf8') as f:f.write('\n'.join(lines))
    print(json.dumps({'status':out['status'],'receipt':str(receipt),'report':str(report),
                      'systemd_next':props['NextElapseUSecRealtime'],'windows_next':record['NextRunTime']}))


if __name__=='__main__':main()
