"""Summarize only verified first-fold minute results from live local queues."""
import csv
from datetime import datetime,timezone
import json

from multiscale.bars import file_digest
from multiscale.spec import MODELS,OUT


def main():
    roots=[OUT/'private_minute_breadth_development/20261107T144302924248Z',
           OUT/'private_minute_breadth_development/20261107T144956615202Z',
           OUT/'private_minute_breadth_development/20261107T150838302360Z',
           OUT/'private_minute_mixed_development/20261107T151427101607Z']
    jobs=[]
    for root in roots:
        state=json.loads((root/'PRIVATE_QUEUE_STATE.json').read_text())
        plan=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
        for job in plan['jobs']:
            if job['fold']!=1:continue
            entry=state['workers'][job['key']]
            if entry['status']!='completed':continue
            folder=root/job['key'];verify_path=root/(job['key']+'_verification.json')
            if not verify_path.is_file():raise ValueError('Completed first fold lacks verification')
            verification=json.loads(verify_path.read_text())
            if verification['fold_receipt_sha256']!=file_digest(folder/'fold_receipt.json'):
                raise ValueError('A first-fold receipt changed')
            receipt=json.loads((folder/'fold_receipt.json').read_text())
            if (receipt['config']['horizon']!=job['horizon'] or receipt['config']['target']!=job['target'] or
                    set(verification['models'])!=set(MODELS)):
                raise ValueError('First-fold task identity differs')
            jobs.append({'horizon':job['horizon'],'target':job['target'],'fold':1,
                         'validation_origins':receipt['validation_origins'],
                         'relative_improvement':{name:verification['models'][name]['against_last_close_or_zero']
                                                 ['relative_improvement'] for name in MODELS},
                         'verification':str(verify_path),'receipt_sha256':file_digest(folder/'fold_receipt.json')})
    jobs.sort(key=lambda x:(x['horizon'],x['target']))
    if len({(j['horizon'],j['target']) for j in jobs})!=len(jobs):raise ValueError('Duplicate first fold')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    parent=OUT/'private_minute_first_fold_reports';parent.mkdir(parents=True,exist_ok=True)
    directory=parent/stamp;directory.mkdir(exist_ok=False)
    result={'status':'private_first_fold_development_snapshot','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'tasks':jobs,'independent_test':False,'publication_rights':'not confirmed',
            'limits':['One fold per displayed task, not the full five-fold result.',
                      'All results are already-exposed development; validation paths overlap.',
                      'One random seed and at most 20 epochs do not establish convergence.']}
    result_path=directory/'PRIVATE_FIRST_FOLDS.json'
    with result_path.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    labels={'last_close':'持续性／零收益','drift':'漂移','seasonal':'季节','ridge':'Ridge',
            'mlp':'MLP','gru':'GRU','lstm':'LSTM','xpatch':'xPatch'}
    lines=['# 分钟任务：已完成首折的本机开发快照','',
           f'生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}。每行仅一折；后续折仍在运行或排队。','',
           '数值为相对持续性价格或零收益基线的终点均方误差改善百分比；正数代表误差较小，负数代表误差较大，不是投资收益率。','',
           '| 任务 | 验证起点 | 漂移 | 季节 | Ridge | MLP | GRU | LSTM | xPatch |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    rows=[]
    for j in jobs:
        target='价格' if j['target']=='future_close' else '累计对数收益'
        values=j['relative_improvement']
        lines.append(f"| H{j['horizon']} {target} | {j['validation_origins']} | "+' | '.join(
            f'{100*values[name]:.2f}%' if values[name] is not None else '不可计算' for name in MODELS[1:])+' |')
        for name in MODELS:
            rows.append({'horizon':j['horizon'],'target':j['target'],'fold':1,'validation_origins':j['validation_origins'],
                         'model':name,'relative_improvement':values[name]})
    lines += ['', '首折、重叠窗口和一种随机种子的结果不能证明长期稳定性，也不能当作独立测试；账号和论文发表权限仍需确认。','',
              '[机器可读结果]('+str(result_path)+')','']
    report=directory/'PRIVATE_FIRST_FOLDS_ZH.md'
    with report.open('x',encoding='utf8') as f:f.write('\n'.join(lines))
    with (directory/'PRIVATE_FIRST_FOLDS.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['horizon','target','fold','validation_origins','model','relative_improvement'])
        w.writeheader();w.writerows(rows)
    print(json.dumps({'status':result['status'],'report':str(report),'result':str(result_path),
                      'tasks':len(jobs),'sha256':file_digest(result_path)},ensure_ascii=False))


if __name__=='__main__':main()
