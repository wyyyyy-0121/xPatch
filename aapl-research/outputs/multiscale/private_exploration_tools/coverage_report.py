"""Create a timestamped, honest snapshot of private development coverage."""
import csv,json
from datetime import datetime,timezone
from pathlib import Path

from multiscale.spec import OUT,tasks
from multiscale.bars import file_digest


def main():
    root=OUT/'private_three_scale_development/20261107T134128870773Z'
    protocol=json.loads((root/'PRIVATE_PROTOCOL.json').read_text());state=json.loads((root/'PRIVATE_QUEUE_STATE.json').read_text())
    controls=OUT/'private_minute_full_controls/20261107T140007359046Z'
    control_check=controls/'PRIVATE_INDEPENDENT_CHECK.json'
    neural=OUT/'private_minute_full_neural/20261107T140317864464Z'
    nstate=json.loads((neural/'PRIVATE_STATE.json').read_text()) if (neural/'PRIVATE_STATE.json').exists() else {}
    ncheck=neural/'PRIVATE_INDEPENDENT_CHECK.json'
    sampled=OUT/'private_minute_exploration/20261107T133934140466Z/PRIVATE_RESULT_SUMMARY.json'
    annual=OUT/'private_annual_short_development/20261107T143734746737Z'
    aprotocol=json.loads((annual/'PRIVATE_PROTOCOL.json').read_text())
    astate=json.loads((annual/'PRIVATE_QUEUE_STATE.json').read_text())
    minute_runs=[OUT/'private_minute_parallel_folds/20261107T142105271032Z',
                 OUT/'private_minute_parallel_folds/20261107T143936100555Z']
    minute_folds={}
    for run in minute_runs:
        run_state=json.loads((run/'PRIVATE_QUEUE_STATE.json').read_text())
        for number,worker in run_state['workers'].items():
            fold=int(number)
            if fold in minute_folds:raise ValueError('Duplicate minute fold')
            verified=(run/f'fold_{fold:02d}_verification.json').is_file()
            minute_folds[fold]='completed_verified' if worker['status']=='completed' and verified else worker['status']
    breadth_runs={5:OUT/'private_minute_breadth_development/20261107T144302924248Z',
                  15:OUT/'private_minute_breadth_development/20261107T144956615202Z',
                  30:OUT/'private_minute_breadth_development/20261107T150838302360Z'}
    breadth_states={h:json.loads((run/'PRIVATE_QUEUE_STATE.json').read_text()) for h,run in breadth_runs.items()}
    mixed=OUT/'private_minute_mixed_development/20261107T151427101607Z'
    mixed_state=json.loads((mixed/'PRIVATE_QUEUE_STATE.json').read_text())
    matrix=OUT/'private_minute_matrix_development/20261107T151850900588Z'
    matrix_plan=json.loads((matrix/'PRIVATE_PROTOCOL.json').read_text())
    matrix_state=json.loads((matrix/'PRIVATE_QUEUE_STATE.json').read_text())
    rows=[]
    for task in tasks():
        profile,h,target=task['profile'],task['horizon'],task['target']
        jobs=[j for j in protocol['jobs'] if (j['profile'],j['horizon'],j['target'])==(profile,h,target)]
        complete=sum(state['jobs'].get(j['key'],{}).get('status')=='completed' for j in jobs)
        details='';scope='original_task_shape_development'
        if jobs:
            status='已完成' if complete==5 else '进行中或未完成'
            details='八类模型；个人历史开发，已逐折复算。' if complete==5 else '以队列状态为准。'
        elif profile=='annual':
            scope='short_history_12_month_lookback_development'
            ajobs=[j for j in aprotocol['jobs'] if j['target']==target]
            complete=sum(astate['jobs'].get(j['key'],{}).get('status')=='completed' and
                         (annual/(j['key']+'.verification.json')).is_file() for j in ajobs)
            status='短历史探索完成' if complete==5 else '短历史探索未完成'
            details='用户选择约10年日线；12个月回看，偏离原定15年方案。五折共47个重叠验证起点。'
        elif profile=='minute' and h in breadth_runs:
            key=f'h{h}_price_f1' if target=='future_close' else f'h{h}_return_f1'
            worker=breadth_states[h]['workers'][key]
            verified=(breadth_runs[h]/(key+'_verification.json')).is_file()
            if worker['status']=='completed' and verified:
                status='首折完成';complete=1;details='完整起点八模型首折已完成并复算；其余四折未完成。'
            elif worker['status']=='running':
                status='首折运行中';details='价格和收益首折并行，尚不计为完整折。'
            elif worker['status']=='paused_night':
                status='暂停至03:00';details='首折已暂停，保留内存状态和磁盘检查点。'
            elif worker['status']=='paused_budget_exhausted':
                status='预算到期暂停';details='首折已暂停并保留检查点，未计为完整折。'
            else:
                status='首折未完成';details='须检查队列状态和模型检查点。'
            if h==5 and target=='future_log_return':
                second=mixed_state['workers']['h5_return_f2']
                second_verified=(mixed/'h5_return_f2_verification.json').is_file()
                if second['status']=='completed' and second_verified:
                    complete+=1;status='两折完成';details='前两折完整起点已复算；其余三折未完成。'
                elif second['status']=='running':
                    status='第二折运行中';details='首折已复算；第二折正在运行，其余折未完成。'
                elif second['status']=='paused_night':
                    status='暂停至03:00';details='首折已复算；第二折已暂停并保留检查点。'
                elif second['status']=='paused_budget_exhausted':
                    status='预算到期暂停';details='首折已复算；第二折预算到期暂停并保留检查点。'
                else:
                    details+=' 第二折未完成，须检查队列状态。'
        elif (profile,h,target)==('minute',60,'future_close'):
            worker=mixed_state['workers']['h60_price_f1']
            verified=(mixed/'h60_price_f1_verification.json').is_file()
            if worker['status']=='completed' and verified:
                status='首折完成';complete=1;details='完整起点八模型首折已完成并复算；其余四折未完成。'
            elif worker['status']=='running':
                status='首折运行中';details='价格首折正在运行，尚不计为完整折。'
            elif worker['status']=='paused_night':
                status='暂停至03:00';details='价格首折已暂停并保留检查点，尚不计为完整折。'
            elif worker['status']=='paused_budget_exhausted':
                status='预算到期暂停';details='价格首折预算到期暂停并保留检查点。'
            else:
                status='首折未完成';details='须检查队列状态和模型检查点。'
        elif (profile,h,target)==('minute',60,'future_log_return'):
            if control_check.exists() and ncheck.exists():
                complete=1+sum(v=='completed_verified' for v in minute_folds.values())
                running=[str(k) for k,v in sorted(minute_folds.items()) if v=='running']
                paused=[str(k) for k,v in sorted(minute_folds.items()) if v=='paused_night']
                status='五折完成' if complete==5 else '进行中'
                details='已完成折均使用完整起点并复算。'
                if running:details+=' 第 '+ '、'.join(running)+' 折正在运行。'
                if paused:status='暂停至03:00';details+=' 第 '+ '、'.join(paused)+' 折已暂停并保留检查点。'
                unfinished=[str(k) for k,v in sorted(minute_folds.items()) if v not in ('running','paused_night','completed_verified')]
                if unfinished:details+=' 第 '+ '、'.join(unfinished)+' 折未完成，须检查队列。'
            elif control_check.exists():
                status='首折部分';details='完整起点四项基线/Ridge已复算；神经模型已完成 '+str(len(nstate.get('completed_models',[])))+'/4。'
            else:status='待运行';details='完整起点对照尚未完成。'
            if sampled.exists():details+=' 另有1/30起点抽样单折，不计为完整折。'
        else:
            status='待运行';details='完整分钟任务及计算预算尚未完成。'
        if profile=='minute':
            external_running=('运行中' in status or status=='进行中')
            mjobs=[j for j in matrix_plan['jobs'] if (j['horizon'],j['target'])==(h,target)]
            added=0;running=[];pending=[];unfinished=[]
            for job in mjobs:
                entry=matrix_state['jobs'][job['key']]
                vp=matrix/'verifications'/(job['key']+'.json')
                if entry['status']=='completed' and vp.is_file() and entry['verification_sha256']==file_digest(vp):
                    added+=1
                elif entry['status']=='running':running.append(job['fold'])
                elif entry['status']=='paused_night':pass
                elif entry['status']=='pending':pending.append(job['fold'])
                else:unfinished.append(job['fold'])
            complete+=added
            if complete>5:raise ValueError('Duplicate minute folds in coverage')
            if mjobs:
                if complete==5:status='五折完成'
                elif running:status='进行中'
                elif matrix_state['status']=='paused_until_0300':status='暂停至03:00'
                elif pending and matrix_state['status'] in ('running','running_night_resume'):status='进行中及排队中' if external_running else '排队中'
                elif matrix_state['status']=='paused_budget_exhausted':status='预算到期暂停'
                else:status='部分完成，需检查队列'
                details+=' 剩余折已纳入两小时队列。'
                if running:details+=' 队列运行第 '+ '、'.join(map(str,running))+' 折。'
                if unfinished:details+=' 队列有未完成折，须检查状态。'
        rows.append({'profile':profile,'horizon':h,'target':target,'protocol_scope':scope,'completed_full_folds':complete,
                     'expected_folds':5,'status':status,'details':details})
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');directory=OUT/'private_coverage';directory.mkdir(parents=True,exist_ok=True)
    cp=directory/(stamp+'.csv');mp=directory/(stamp+'.md')
    with cp.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    labels={'annual':'年度','quarterly':'季度','monthly':'月度','hourly':'小时','minute':'分钟'}
    text=['# AAPL 多尺度内部实验覆盖','',f'生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}。这是个人历史开发进度；发表许可和独立确认均未完成。','',
          '| 尺度 | 跨度 | 目标 | 完整起点折数 | 状态 | 说明 |','| --- | ---: | --- | ---: | --- | --- |']
    for r in rows:
        target='价格' if r['target']=='future_close' else '累计对数收益'
        text.append(f"| {labels[r['profile']]} | H{r['horizon']} | {target} | {r['completed_full_folds']}/5 | {r['status']} | {r['details']} |")
    text += ['', '分钟抽样单折不计入完整起点折数；年度为用户同意的短历史探索，不能称为原定15年方案的完成。季度和年度验证起点相互重叠；模型结果、供应商许可及数据口径仍须按实际情况复核。','',
             '[配套 CSV]('+str(cp)+')','']
    with mp.open('x',encoding='utf8') as f:f.write('\n'.join(text))
    print(json.dumps({'status':'coverage_snapshot_created','path':str(mp),'csv':str(cp),'tasks':len(rows),
                      'development_five_fold_tasks':sum(r['completed_full_folds']==5 for r in rows),
                      'original_annual_protocol_satisfied':False,
                      'sha256':file_digest(mp)},ensure_ascii=False))


if __name__=='__main__':main()
