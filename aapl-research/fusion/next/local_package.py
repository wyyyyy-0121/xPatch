"""Build an immutable CPU delivery with verified code, fitted models and a known-period example."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile

import pandas as pd

from .common import ROOT,OUT,PLAN,digest,identity,read_json,write_json,utc_now,source_files,source_identity
from .foundation import manifest as weight_manifest,weight_directory
from .inference import checked_manifest


RUNTIME_PACKAGES=('torch','numpy','pandas','scipy','scikit-learn','lightgbm','chronos-forecasting',
                  'transformers','peft','accelerate','safetensors','exchange-calendars','psutil')


def replay_example(state,model_identity):
    """Only the last retained known-period context; never acquire new market values."""
    if state.get('schema')!=2 or state.get('integrity')!=identity({k:v for k,v in state.items() if k!='integrity'}):
        raise ValueError('Acceptance state checksum differs')
    if state['recipe']['model_identity']!=model_identity:raise ValueError('Example model differs')
    outputs=state['outputs']
    if not outputs:raise ValueError('Acceptance has no completed predictions')
    origin=max(outputs);expected=outputs[origin]
    if expected['status']!='ok' or expected['mode']!='delayed_historical_replay':raise ValueError('Example requires a successful delayed replay')
    if expected['model_identity']!=model_identity:raise ValueError('Example prediction model differs')
    events=[]
    for symbol,bars in state['bars'].items():
        if symbol not in list(dict.fromkeys(['AAPL']+state['recipe']['background'])):raise ValueError('Unexpected example symbol')
        for stamp,row in bars.items():
            date=pd.Timestamp(stamp).tz_convert('America/New_York').strftime('%Y-%m-%d')
            if not '2026-01-01'<=date<='2026-03-31':raise ValueError('Example values must be in the known Q1 development period')
            if pd.Timestamp(stamp)>pd.Timestamp(origin):raise ValueError('Example contains values after its origin')
            events.append({'type':'bar','symbol':symbol,'bar_start':stamp,'received_at':row['received_at'],'values':row['values']})
    if not events:raise ValueError('Example has no input bars')
    events.sort(key=lambda row:(row['bar_start'],row['symbol']))
    dates=[pd.Timestamp(e['bar_start']).tz_convert('America/New_York').strftime('%Y-%m-%d') for e in events]
    events.append({'type':'predict','bar_start':origin,'generated_at':expected['generated_at']})
    description={'calendar_start':str((pd.Timestamp(min(dates))-pd.Timedelta(days=7)).date()),
                 'calendar_end':str((pd.Timestamp(max(dates))+pd.Timedelta(days=7)).date()),
                 'mode':'delayed_historical_replay','expected_prediction':expected['predicted_log_return'],
                 'model_identity':model_identity,'origin':origin,'values_period':'known 2026Q1 development only'}
    return events,description


def build(deployment):
    deployment=Path(deployment).resolve();model=checked_manifest(deployment)
    freeze=OUT/'system_freeze.json';frozen=read_json(freeze)
    if digest(freeze)!=model['freeze_sha256']:raise ValueError('Deployment freeze differs')
    if frozen['source_identity']!=source_identity() or frozen['plan_sha256']!=digest(PLAN):raise ValueError('Delivery code differs from the frozen research')
    acceptance=OUT/'deployment_acceptance'/model['identity']
    report=read_json(acceptance/'report.json');progress=read_json(acceptance/'progress.json')
    if report['model_identity']!=model['identity'] or report['Q3_accessed']:raise ValueError('Acceptance identity or example period differs')
    if progress['recipe']['source_identity']!=source_identity() or progress['recipe']['system_freeze_sha256']!=digest(freeze):
        raise ValueError('Acceptance used different code or a different frozen system')
    events,example=replay_example(read_json(acceptance/'state.json'),model['identity'])
    packages={name:metadata.version(name) for name in RUNTIME_PACKAGES}
    items=source_files();items['FUSION_IMPLEMENTATION_PLAN.md']=PLAN
    items['deployment/deployment.json']=deployment/'deployment.json'
    for name in model['files']:items['deployment/'+name]=deployment/name
    for name in ('system_freeze.json','confirmation.json'):
        items['evidence/'+name]=OUT/name
    items['evidence/deployment_acceptance.json']=acceptance/'report.json'
    needs_foundation=any(entry['config'].get('foundation',True) for entry in model['system_checkpoints'])
    if needs_foundation:
        weights=weight_manifest();directory=weight_directory()
        items['outputs/fusion_next/foundation_weights.json']=OUT/'foundation_weights.json'
        for name,row in weights['files'].items():
            path=(directory/name).resolve()
            if not path.is_relative_to(directory.resolve()) or digest(path)!=row['sha256']:raise ValueError('Pinned foundation weight differs')
            items['weights/chronos2/'+name]=path
    target=OUT/'local_packages'/model['identity'];target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() or target.with_suffix('.tar').exists():raise ValueError('Delivery already exists; preserve its original evidence')
    with tempfile.TemporaryDirectory(prefix='delivery_build_',dir=ROOT/'tmp') as temporary:
        root=Path(temporary)/'aapl-cpu';root.mkdir()
        for name,path in items.items():
            destination=root/name
            if not destination.resolve().is_relative_to(root):raise ValueError('Delivery path escapes its root')
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
        write_json(root/'runtime_lock.json',{'python':platform.python_version(),'packages':packages,
                   'all_installed_distributions':{d.metadata['Name']:d.version for d in metadata.distributions() if d.metadata.get('Name')}})
        (root/'requirements.txt').write_text('\n'.join(name+'=='+version for name,version in packages.items())+'\n')
        shutil.copyfile(ROOT/'fusion/next/cloud_assets/cpu_run.py',root/'run.py')
        (root/'example.jsonl').write_text(''.join(json.dumps(event,allow_nan=False)+'\n' for event in events))
        write_json(root/'example.json',example)
        (root/'README.md').write_text(documentation(model,report,example),encoding='utf-8')
        files={str(p.relative_to(root)):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(root.rglob('*')) if p.is_file()}
        receipt={'schema':1,'model_identity':model['identity'],'source_identity':source_identity(),'plan_sha256':digest(PLAN),
                 'created_at_utc':utc_now(),'files':files,'foundation_weights_included':needs_foundation,
                 'example_period':'known 2026Q1 only','credentials_included':False,'Q3_values_included':False,
                 'engineering_acceptance_passed':report['engineering_acceptance_passed']}
        receipt['identity']=identity(receipt);write_json(root/'package.json',receipt)
        shutil.copytree(root,target)
    archive=target.with_suffix('.tar')
    with tarfile.open(archive,'w') as stream:
        for path in sorted(target.rglob('*')):
            if path.is_file():stream.add(path,arcname='aapl-cpu/'+str(path.relative_to(target)),recursive=False)
    write_json(target.parent/(model['identity']+'.receipt.json'),{'archive_sha256':digest(archive),'archive_bytes':archive.stat().st_size,
               'package_identity':receipt['identity'],'model_identity':model['identity'],'created_at_utc':utc_now()})
    return archive


def documentation(model,report,example):
    return f'''# AAPL CPU 推理包

固定系统身份：`{model['identity']}`。输出为未来 12 分钟累计 log 收益。
工程验收通过：`{report['engineering_acceptance_passed']}`；预测研究是否达标见 `evidence/confirmation.json`。
工程验收结果与预测改善分别判断。包内没有交易执行功能。

在 Linux/WSL 的独立 Python 环境安装 `requirements.txt`，PyTorch 使用与本地验收一致的版本。
Python 和完整环境版本见 `runtime_lock.json`。不复制虚拟环境；不修改全局配置。

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py --verify-only
.venv/bin/python run.py --example
```

示例只含已知 2026Q1 的历史上下文，接收/生成时间为模拟重放时间。
首次示例预测与保存结果数值核对，重复执行自动恢复同一状态与预测记录。
示例日历范围：{example['calendar_start']} 至 {example['calendar_end']}。
运行入口默认使用 4 个可用逻辑 CPU，并限制在原研究允许的 CPU 0—17 内；可通过 AAPL_CPU_IDS/AAPL_CPU_THREADS 设置其中的子集。

运行自己的完成分钟柱输入：

```bash
.venv/bin/python run.py --calendar-start 2026-10-01 --calendar-end 2026-10-31 --state outputs/feed/state.json < feed.jsonl
```

每行一个 JSON 事件，包含显式时区。输入股票为 AAPL 及固定背景：{', '.join(model['background'])}。
bar 事件包含 `type=bar`、`symbol`、`bar_start`、`received_at`、`values`；values 必含 open/high/low/close/volume。
predict 事件包含 `type=predict`、`bar_start`，可含 `generated_at`。
按时间送入至少 242 根已完成常规时段柱，包括前一交易日历史；日历范围应覆盖这些历史及当日预测终点。
默认 `delayed_historical_replay`，在线及时预测须显式设置 `--mode timely_forecast` 并提供真实接收/生成时间。
通用接口拒绝研究保留期 2026Q3 的输入；该时段只经研究专用最终检验入口执行。

标准输出为逐事件 JSON；预测包含模型身份、输入截止、生成时间、目标时间、收益和专家权重。
有输入拒绝或模型失败时返回非零退出码，并保留失败记录。恢复已有状态加 `--resume`。
预测只使用已接收输入，不需要目标价格；需要自己提供合法的行情源。
'''


def verify(archive):
    """Actually unpack and execute the CPU example in another process."""
    archive=Path(archive).resolve()
    with tempfile.TemporaryDirectory(prefix='cpu_package_verify_',dir=ROOT/'tmp') as temporary:
        directory=Path(temporary)
        with tarfile.open(archive) as stream:
            seen=set()
            for member in stream.getmembers():
                if not member.isfile() or member.name in seen or not member.name.startswith('aapl-cpu/'):
                    raise ValueError('CPU archive contains an unexpected member')
                if not (directory/member.name).resolve().is_relative_to(directory):raise ValueError('CPU archive leaves isolated directory')
                seen.add(member.name)
            stream.extractall(directory,filter='data')
        isolated=directory/'aapl-cpu';env=dict(os.environ,PYTHONPATH=str(isolated),PYTHONDONTWRITEBYTECODE='1')
        results=[]
        for args in (['--verify-only'],['--example'],['--example']):
            result=subprocess.run([sys.executable,'run.py',*args],cwd=isolated,env=env,capture_output=True,text=True,timeout=180)
            if result.returncode:raise RuntimeError('Isolated CPU delivery failed: '+result.stderr[-3000:])
            results.append({'arguments':args,'stdout':result.stdout[-6000:],'stderr':result.stderr[-3000:]})
        receipt=read_json(isolated/'package.json')
    report={'archive_sha256':digest(archive),'package_identity':receipt['identity'],'model_identity':receipt['model_identity'],
            'files_hash_verified':len(receipt['files']),'isolated_example_and_fresh_process_resume_passed':True,
            'Q3_accessed':False,'processes':results,'verified_at_utc':utc_now()}
    write_json(archive.with_suffix('.verification.json'),report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--deployment',type=Path,required=True)
    parser.add_argument('--verify',action='store_true');args=parser.parse_args();archive=build(args.deployment)
    print(verify(archive) if args.verify else archive)
