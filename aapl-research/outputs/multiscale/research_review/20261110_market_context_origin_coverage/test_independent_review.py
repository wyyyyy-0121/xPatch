"""New independent-implementation checks only, using generated values/files."""
import ast
from datetime import date,datetime,time,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import numpy as np
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('independent_test',BASE/'independent_coverage_review.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert os.sched_getaffinity(0)=={26}
checks=[]
def check(name,value):
    assert value,name
    checks.append(name)
def reject(name,call):
    try:call()
    except m.ReviewError:checks.append(name)
    else:raise AssertionError(name)
imports=[n.module for n in ast.walk(ast.parse((BASE/'independent_coverage_review.py').read_text())) if isinstance(n,ast.ImportFrom)]
imports += [alias.name for n in ast.walk(ast.parse((BASE/'independent_coverage_review.py').read_text())) if isinstance(n,ast.Import) for alias in n.names]
check('no_primary_core_worker_or_source_import',all('coverage_core' not in name and 'coverage_worker' not in name and 'download_spy' not in name for name in imports))
source=np.array([True,False,True,True,False,True,True])
check('independent_sliding_window_exact_excludes_current',m.history_mask_all(source,2).tolist()==[False,False,False,False,True,False,False])
check('insufficient_lower_history_kept_missing',m.history_mask_all(source,8).tolist()==[False]*7)
ends=np.array([1,2,3,4,5,6,90,91,92,93,94],dtype=np.int64)
cut,prefix,tail=m.independent_partition(np.arange(9,dtype=np.int64),ends,2)
check('independent_explicit_clock_purge_and_untouched_tail',cut==7 and prefix.tolist()==[0,1,2,3,4] and tail.tolist()==[7,8])
start=np.r_[np.arange(72),1440+np.arange(61)].astype(np.int64)*m.MINUTE
sessions=np.array(['2021-01-04']*72+['2021-01-05']*61)
minute={'start_ns':start,'end_ns':start+m.MINUTE,'session_date':sessions}
hour={'start_ns':start[[0,72]],'end_ns':(start+m.MINUTE)[[59,131]],'session_date':sessions[[0,72]]}
present=np.ones(133,dtype=bool);present[65]=False;present[80]=False
check('independent_sessions_residuals_and_constituent_missing',m.independent_hours(minute,hour,present).tolist()==[True,False])
wrong=dict(hour,start_ns=hour['start_ns'].copy());wrong['start_ns'][1]=start[60]
reject('independent_inbounds_cross_session_mismatch',lambda:m.independent_hours(minute,wrong,present))
reject('explicit_member_allowlist_precedes_open',lambda:m.selected_npz('/no/file',('close',),[],lambda:None))
with tempfile.TemporaryDirectory(prefix='new_independent_clock_') as directory:
    directory=Path(directory)
    index={'kind':'SPY_qualified_canonical_index_v1','all_67_ranges_complete':True,'ranges':m.fixed_ranges(),
           'Q3_allowed':False,'source_only_not_RTH_or_context_admission':True,'protocol_sha256':'artificial',
           'inheritance_manifest_sha256':'artificial','pages':[]}
    for number,(first,last) in enumerate(m.fixed_ranges()):
        stamp=datetime.combine(date.fromisoformat(first),time(),m.NY).astimezone(timezone.utc).isoformat()
        body=json.dumps({'bars':{'SPY':[{'t':stamp,'o':1,'h':1,'l':1,'c':1,'v':0,'n':0,'vw':1}]},'next_page_token':None}).encode()
        path=directory/f'artificial_{number}.json';path.write_bytes(body)
        index['pages'].append({'range':number,'page':0,'canonical_ref':{'path':str(path),'bytes':len(body),'sha256':m.sha(body)},
             'request_page_token':None,'next_page_token':None,'rows':1,'first_UTC':stamp,'last_UTC':stamp})
    values,refs=m.independent_source(index,{'source_protocol_sha256':'artificial','inheritance_manifest_sha256':'artificial'},lambda:None)
    check('independent_67_range_clock_reader_artificial_only',len(values)==67 and len(refs)==67 and np.all(values[1:]>values[:-1]))
    index['pages'][0]['canonical_ref']['sha256']='0'*64
    reject('independent_reads_verify_exact_payload_hash',lambda:m.independent_source(index,{'source_protocol_sha256':'artificial','inheritance_manifest_sha256':'artificial'},lambda:None))
    path=directory/'false_protocol.json';path.write_text('{"kind":"SPY_original_origin_clock_coverage_v1","execution_admitted":false}')
    reject('false_protocol_rejects_before_real_inputs',lambda:m.run(path))
receipt={'status':'new_independent_implementation_checks_only_not_actual_execution','checks_count':len(checks),'checks':checks,
         'CPU_affinity':sorted(os.sched_getaffinity(0)),'actual_market_raw_or_arrays_read':False,'actual_prices_labels_or_Q3_read':False,'HTTP_calls':0,
         'primary_existing_fixtures_not_rerun':True,'independent_reviewer_sha256':hashlib.sha256((BASE/'independent_coverage_review.py').read_bytes()).hexdigest(),
         'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(BASE/'INDEPENDENT_SYNTHETIC_CHECKS.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'checks':len(checks),'independent_reviewer_sha256':receipt['independent_reviewer_sha256']}))
