"""Actual runtime fingerprint and process limits; no model downloads or training."""
import argparse
import importlib.metadata as metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import psutil
import torch

from .common import OUT, PLAN, digest, utc_now, write_json


def runtime(seed=2026):
    import random
    import numpy as np
    profile=os.environ.get('FUSION_EXECUTION_PROFILE','local')
    affinity=sorted(os.sched_getaffinity(0))
    if profile!='autodl' and len(affinity)>18:
        raise RuntimeError('Local research must use scripts/python.sh with the shared <=18 CPU allocation')
    threads=int(os.environ.get('AAPL_CPU_THREADS',os.environ.get('OMP_NUM_THREADS','4')))
    if threads>len(affinity):raise RuntimeError('Thread pool exceeds process CPU affinity')
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if profile=='autodl':
        if not torch.cuda.is_available():raise RuntimeError('Cloud training requires a working CUDA runtime')
        torch.cuda.manual_seed_all(seed)
        return 'cuda'
    return 'cpu'  # Large neural experiments have explicitly moved to the user's cloud GPU.


def inspect_hardware(require_cuda=False):
    packages={}
    for name in ('torch','numpy','pandas','scipy','scikit-learn','lightgbm',
                 'chronos-forecasting','transformers','peft','accelerate','safetensors','exchange-calendars','psutil'):
        try:packages[name]=metadata.version(name)
        except metadata.PackageNotFoundError:packages[name]=None
    memory=psutil.virtual_memory();disk=shutil.disk_usage(OUT.parent)
    report={'recorded_at_utc':utc_now(),'profile':os.environ.get('FUSION_EXECUTION_PROFILE','local'),
            'python':platform.python_version(),'platform':platform.platform(),
            'cpu_affinity':sorted(os.sched_getaffinity(0)),'logical_cpu_count':psutil.cpu_count(),
            'ram_total_bytes':memory.total,'ram_available_bytes':memory.available,
            'disk_free_bytes':disk.free,'runtime_cuda':torch.version.cuda,'packages':packages,
            'plan_sha256':digest(PLAN),'gpu':None,'F2_performance_admitted':False}
    report['installed_distributions']={d.metadata['Name']:d.version for d in metadata.distributions() if d.metadata.get('Name')}
    for name in ('memory.max','cpu.max'):
        path=Path('/sys/fs/cgroup')/name
        if path.is_file():report['cgroup_'+name]=path.read_text().strip()
    if require_cuda:
        if not torch.cuda.is_available():raise RuntimeError('CUDA not available')
        if torch.cuda.device_count()!=1:raise RuntimeError('This budget is reserved for exactly one GPU')
        device=torch.cuda.get_device_properties(0)
        start=time.perf_counter()
        x=torch.randn(1024,1024,device='cuda');value=(x@x).sum();torch.cuda.synchronize()
        if not torch.isfinite(value):raise RuntimeError('Actual CUDA matrix operation was nonfinite')
        report['gpu']={'name':device.name,'vram_bytes':device.total_memory,
                       'compute_capability':[device.major,device.minor],
                       'built_arches':torch.cuda.get_arch_list(),'actual_kernel_passed':True,
                       'kernel_seconds':time.perf_counter()-start,
                       'bf16_supported':torch.cuda.is_bf16_supported()}
        del x,value
        try:
            report['nvidia_smi']=subprocess.check_output(
                ['nvidia-smi','--query-gpu=name,driver_version,memory.total,temperature.gpu,clocks.sm,power.draw',
                 '--format=csv,noheader'],text=True,timeout=10).strip()
        except (OSError,subprocess.SubprocessError):report['nvidia_smi']=None
    return report


class ResourceGuard:
    def __init__(self,device='cpu',rss_gib=8.,available_gib=4.,gpu_fraction=.85):
        self.device=device;self.rss_limit=rss_gib*2**30;self.available_limit=available_gib*2**30
        self.gpu_fraction=gpu_fraction;self.peak_rss=0;self.peak_vram=0;self.minimum_available=None

    def check(self):
        process=psutil.Process();rss=process.memory_info().rss
        available=psutil.virtual_memory().available
        self.peak_rss=max(self.peak_rss,rss)
        self.minimum_available=available if self.minimum_available is None else min(self.minimum_available,available)
        if rss>self.rss_limit:raise MemoryError('Process RSS exceeds admitted limit')
        if available<self.available_limit:raise MemoryError('System available RAM below reserved minimum')
        if self.device=='cuda':
            free,total=torch.cuda.mem_get_info();used=max(total-free,torch.cuda.max_memory_reserved())
            self.peak_vram=max(self.peak_vram,used)
            if used>self.gpu_fraction*total:raise MemoryError('Device VRAM exceeds 85% admission limit')
        return {'peak_rss_bytes':self.peak_rss,'minimum_available_ram_bytes':self.minimum_available,
                'peak_device_used_vram_bytes':self.peak_vram}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--require-cuda',action='store_true');args=parser.parse_args()
    report=inspect_hardware(args.require_cuda or os.environ.get('FUSION_EXECUTION_PROFILE')=='autodl')
    write_json(OUT/('hardware_cloud.json' if report['gpu'] else 'hardware_local.json'),report)
    print(report)
