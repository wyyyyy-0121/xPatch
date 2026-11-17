"""Concurrent CPU and GPU sections must not double-count one process's CPU time."""
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest

from . import accounting
from .common import OUT,write_json


def test_CPU_time_union_handles_overlap_and_PID_reuse(monkeypatch):
    with TemporaryDirectory(prefix='unit_cpu_accounting_',dir=OUT) as temporary:
        root=Path(temporary);monkeypatch.setattr(accounting,'OUT',root)
        for index,(start,end,birth) in enumerate([(10.,20.,100.),(15.,25.,100.),(30.,35.,100.),(10.,12.,200.)]):
            write_json(root/'usage_events'/f'{index}.json',{'gpu_card_hours':0.,'category':'experts_and_cache',
                'status':'completed','id':str(index),'boot_id':'toy_boot','pid':1,'process_created_unix':birth,
                'process_cpu_start_seconds':start,'process_cpu_end_seconds':end})
        totals=accounting.totals()
        assert totals['accounted_process_CPU_seconds']==22. and totals['CPU_jobs_without_measurements']==0


def test_user_removed_time_caps_but_single_live_GPU_job_is_still_enforced(monkeypatch):
    with TemporaryDirectory(prefix='unit_unlimited_accounting_',dir=OUT) as temporary:
        root=Path(temporary);monkeypatch.setattr(accounting,'OUT',root)
        write_json(root/'usage_events'/'old.json',{'gpu_card_hours':25.,'category':'performance',
            'status':'completed','id':'old','wall_seconds':90000.})
        with accounting.Usage('performance','authorized work after the former time caps') as job:
            row=job.record();assert row['status']=='running'
            assert accounting.totals()['gpu_card_hours']>=25.
            with pytest.raises(RuntimeError,match='already active'):
                with accounting.Usage('fusion_search','duplicate GPU allocation'):pass
        assert accounting.totals()['active_gpu_jobs']==[]
        assert accounting.totals()['compute_policy']['total_gpu_card_hours_limit'] is None
