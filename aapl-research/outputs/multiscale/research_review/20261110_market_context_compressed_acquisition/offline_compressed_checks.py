"""Only constant artificial bars, fake opener and temporary files. No live HTTP/auth."""
from pathlib import Path
from datetime import timedelta
import concurrent.futures
import copy
import email.message
import gzip
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import threading
import zlib

BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('compressed_test',BASE/'download_compressed_spy.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
assert os.sched_getaffinity(0)=={26}
checks=[]; opens=0

def good(name,condition):
    assert condition,name
    checks.append({'name':name,'passed':True})

def rejects(name,call):
    try:
        call()
    except (m.Halt,m.SOURCE.Halt,m.TRANSPORT.Halt,zlib.error):
        good(name,True)
    else:
        raise AssertionError(name)

def body(index,minute=0,token=None):
    t=m.SOURCE.bounds(index)[0]+timedelta(minutes=minute)
    bar={'t':t.isoformat(),'o':1.0,'h':1.0,'l':1.0,'c':1.0,'v':0,'n':0,'vw':1.0}
    return json.dumps({'bars':{'SPY':[bar]},'next_page_token':token},separators=(',',':')).encode()

class Clock:
    def __init__(self): self.now=1000000.0
    def __call__(self): return self.now

class Response(io.BytesIO):
    def __init__(self,data,encoding='gzip',callback=None):
        super().__init__(data); self.status=200; self.url=m.SOURCE.ENDPOINT
        self.headers=email.message.Message(); self.headers['Content-Encoding']=encoding
        self.callback=callback
    def read(self,size):
        result=super().read(size)
        if self.callback:
            callback,self.callback=self.callback,None; callback()
        return result

class Opener:
    def __init__(self,data,encoding='gzip',callback=None):
        self.data=data; self.encoding=encoding; self.callback=callback
    def open(self,request,timeout):
        global opens
        opens+=1
        assert timeout==15 and request.get_header('Accept-encoding')=='gzip'
        return Response(self.data,self.encoding,self.callback)

m.build_opener=lambda *_: (_ for _ in ()).throw(AssertionError('actual_opener_forbidden'))
with tempfile.TemporaryDirectory(prefix='artificial_compressed_source_') as temporary:
    task=Path(temporary); inherited=task/'inherited'; inherited.mkdir()
    prefix=[]; entities=[]
    for number in range(2):
        data=body(0,number,'artificial-next' if number==0 else None)
        file=inherited/f'prefix_{number}.json'; file.write_bytes(data)
        ref={'path':str(file),'bytes':len(data),'sha256':m.digest(data),'kind':'range0_canonical'}
        entities.append(ref); prefix.append(dict(ref,page=number,params=m.SOURCE.parameters(0,None if number==0 else 'artificial-next')))
    wire=gzip.compress(body(0),mtime=0); file=inherited/'old_probe.wire'; file.write_bytes(wire)
    entities.append({'path':str(file),'bytes':len(wire),'sha256':m.digest(wire),'kind':'gzip_probe_wire'})
    filler=b'0'*(2127635-sum(e['bytes'] for e in entities)); file=inherited/'single_day_artificial.bin'; file.write_bytes(filler)
    entities.append({'path':str(file),'bytes':len(filler),'sha256':m.digest(filler),'kind':'single_session_canonical'})
    evidence=[]
    for number in range(8):
        file=inherited/f'evidence_{number}.json'; file.write_text('{"artificial":true}')
        evidence.append({'path':str(file),'sha256':m.digest(file.read_bytes())})
    manifest=task/'manifest.json'; manifest.write_text(json.dumps({'kind':'SPY_compressed_continuation_inheritance_v1',
        'prior_actual_GET':4,'prior_payload_bytes':2127635,'entities':entities,'prefix_pages':prefix,'evidence':evidence,
        'old_windows_closed_not_reset':True}))
    serial=0
    def instance():
        global serial
        serial+=1; directory=task/f'run_{serial}'; directory.mkdir(); clock=Clock()
        protocol=dict(m.template(manifest),execution_admitted=True,started_epoch=clock.now,deadline_epoch=clock.now+3600)
        file=directory/'protocol.json'; file.write_text(json.dumps(protocol))
        a=m.Acquisition(file,root=directory/'acquisition',manifest_path=manifest,clock=clock)
        a.recover(); return a,file,clock
    def reload(a,file,clock):
        return m.Acquisition(file,root=a.root,manifest_path=manifest,clock=clock)
    def fetch(a,attempt,data=None,encoding='gzip',callback=None):
        data=gzip.compress(body(attempt['range']),mtime=0) if data is None else data
        m.build_opener=lambda *_:Opener(data,encoding,callback)
        try:
            a.fetch_wire(attempt,('artificial-key','artificial-secret'))
        finally:
            m.build_opener=lambda *_: (_ for _ in ()).throw(AssertionError('actual_opener_forbidden'))
    def wire_only():
        a,file,clock=instance(); attempt=a.reserve(1,None); fetch(a,attempt)
        return a,file,clock,attempt
    a,file,clock=instance()
    good('carry_in_4GET_and_2127635_bytes',a.state['global_GET_charged']==4 and a.charged_bytes()==2127635)
    good('range0_old_refs_two_pages_no_GET',a.state['ranges'][0]['complete'] and len(a.state['ranges'][0]['pages'])==2 and opens==0)
    rejects('range0_never_reserved',lambda:a.reserve(0,None))
    attempt=a.reserve(1,None)
    good('atomic_double_layer_reservation',a.state['global_GET_charged']==5 and a.charged_bytes()==2127635+m.ATTEMPT_RESERVE)
    fetch(a,attempt)
    good('wire_settles_actual_canonical_stays_reserved',attempt['wire_charged']==a.artifact(attempt,'wire').stat().st_size and attempt['canonical_charged']==m.SIDE_RESERVE)
    a.canonical_from_wire(attempt); a.qualify(attempt)
    good('both_complete_actual_storage_sum',a.charged_bytes()==2127635+attempt['wire_meta']['bytes']+attempt['canonical_meta']['bytes'])
    good('qualified_ref_hash_bytes_token_interface',a.state['ranges'][1]['pages'][0]['canonical_ref']['sha256']==m.digest(body(1)) and a.state['ranges'][1]['pages'][0]['request_page_token'] is None)
    rejects('no_page_after_terminal',lambda:a.reserve(1,None))
    clock.now+=2; a.record_calibration(); frozen=copy.deepcopy(a.state['calibration_cost']); a.continuation_gate(); a.save()
    good('gate_uses_65ranges_and_two_double_layer_safety',a.state['continuation_gate']['remaining_GET_predicted']==98 and m.GATE['inflight_safety_bytes']==16777220)
    before=opens; b=reload(a,file,clock); b.recover()
    good('new_PID_recovery_no_GET_and_frozen_cost',opens==before and b.state['calibration_cost']==frozen and b.state['global_GET_charged']==5)
    a,file,clock,attempt=wire_only(); before=opens; b=reload(a,file,clock); b.recover()
    good('wire_complete_missing_canonical_rebuild_no_GET',opens==before and b.state['attempts'][0]['status']=='qualified')
    a,file,clock,attempt=wire_only(); canon=a.artifact(attempt,'canonical'); partial=canon.with_suffix(canon.suffix+'.partial'); partial.write_bytes(body(1)[:50])
    before=opens; b=reload(a,file,clock); b.recover()
    good('matching_partial_prefix_appended_no_copy_or_GET',opens==before and canon.read_bytes()==body(1) and not partial.exists())
    a,file,clock,attempt=wire_only(); canon=a.artifact(attempt,'canonical'); partial=canon.with_suffix(canon.suffix+'.partial'); partial.write_bytes(b'wrong-prefix')
    before=opens; b=reload(a,file,clock)
    rejects('mismatched_partial_preserved',b.recover)
    good('mismatched_partial_unchanged_and_no_GET',partial.read_bytes()==b'wrong-prefix' and opens==before)
    a,file,clock,attempt=wire_only(); wire=a.artifact(attempt,'wire'); wire.replace(wire.with_suffix('.wire.partial'))
    before=opens; b=reload(a,file,clock); b.recover()
    good('precommitted_wire_temp_promoted_without_GET',wire.exists() and opens==before)
    a,file,clock,attempt=wire_only(); a.canonical_from_wire(attempt); canonical=a.artifact(attempt,'canonical'); canonical.replace(canonical.with_suffix(canonical.suffix+'.partial'))
    before=opens; b=reload(a,file,clock); b.recover()
    good('precommitted_canonical_temp_promoted_without_GET',canonical.exists() and opens==before)
    a,file,clock=instance(); attempt=a.reserve(1,None); partial=a.artifact(attempt,'wire').with_suffix('.wire.partial'); partial.write_bytes(b'incomplete')
    before=opens; b=reload(a,file,clock)
    rejects('unknown_old_request_not_retried',b.recover)
    good('unknown_partial_and_worst_reservation_preserved',opens==before and partial.read_bytes()==b'incomplete' and b.charged_bytes()==2127635+m.ATTEMPT_RESERVE)
    a,file,clock=instance(); attempt=a.reserve(1,None); broken=bytearray(gzip.compress(body(1),mtime=0)); broken[-8]^=1; fetch(a,attempt,bytes(broken))
    rejects('CRC_failure_keeps_canonical_reserved',lambda:a.canonical_from_wire(attempt))
    good('CRC_failure_wire_preserved_and_double_ledger',a.artifact(attempt,'wire').exists() and attempt['canonical_charged']==m.SIDE_RESERVE)
    a,file,clock=instance(); attempt=a.reserve(1,None)
    rejects('identity_response_no_fallback_GET',lambda:fetch(a,attempt,body(1),'identity'))
    good('identity_keeps_request_and_reservations',a.state['global_GET_charged']==5 and a.charged_bytes()==2127635+m.ATTEMPT_RESERVE)
    a,file,clock=instance(); attempt=a.reserve(1,None)
    rejects('wire_guard_excess_preserved',lambda:fetch(a,attempt,b'x'*(m.CAP+1)))
    good('wire_guard_counts_against_reserved_side',a.artifact(attempt,'wire').with_suffix('.wire.partial').stat().st_size==m.CAP+1)
    a,file,clock=instance(); attempt=a.reserve(1,None)
    fetch(a,attempt,gzip.compress(body(1,0,'artificial-page2'),mtime=0)); a.canonical_from_wire(attempt); a.qualify(attempt)
    rejects('wrong_next_request_token_rejected',lambda:a.reserve(1,'wrong'))
    second=a.reserve(1,'artificial-page2'); fetch(a,second,gzip.compress(body(1,1),mtime=0)); a.canonical_from_wire(second); a.qualify(second)
    good('two_page_chain_qualified_and_terminal',a.state['ranges'][1]['complete'] and len(a.state['ranges'][1]['pages'])==2)
    a,file,clock=instance(); attempt=a.reserve(1,None); fetch(a,attempt); wire=a.artifact(attempt,'wire'); wire.with_suffix('.wire.partial').write_bytes(wire.read_bytes())
    rejects('unaccounted_duplicate_payload_blocks_next_GET',lambda:a.reserve(2,None))
    a,file,clock=instance(); old_bytes=a.path.read_bytes(); clock.now+=3600
    rejects('expired_resume_rejected_before_journal_write',lambda:reload(a,file,clock))
    good('expired_resume_journal_byte_identical',a.path.read_bytes()==old_bytes)
    a,file,clock=instance(); protocol=json.loads(file.read_bytes()); protocol['deadline_epoch']+=1; file.write_text(json.dumps(protocol))
    rejects('deadline_cannot_reset_or_extend',lambda:reload(a,file,clock))
    a,file,clock,attempt=wire_only(); wire=a.artifact(attempt,'wire'); canonical=a.artifact(attempt,'canonical'); partial=canonical.with_suffix(canonical.suffix+'.partial'); partial.write_bytes(body(1))
    a.seal('artificial_terminal'); sealed=a.path.read_bytes()
    rejects('sealed_canonical_commit_cannot_rename',lambda:a.commit(attempt,'canonical',{'bytes':len(body(1)),'sha256':m.digest(body(1))}))
    good('sealed_journal_stable_and_complete_temp_retained',a.path.read_bytes()==sealed and partial.exists() and not canonical.exists())
    a,file,clock=instance(); attempt=a.reserve(1,None); before=opens
    rejects('late_wire_worker_cannot_commit_after_seal',lambda:fetch(a,attempt,callback=lambda:a.seal('artificial_deadline')))
    good('late_wire_preserved_partial_without_final_or_journal_mutation',not a.artifact(attempt,'wire').exists() and a.artifact(attempt,'wire').with_suffix('.wire.partial').exists() and opens==before+1)
    a,file,clock=instance(); a.prior_bytes=268435456-m.ATTEMPT_RESERVE
    outcomes=[]
    def reserve_race(index):
        try:
            a.reserve(index,None); outcomes.append('reserved')
        except m.Halt:
            outcomes.append('blocked')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(reserve_race,(1,2)))
    good('two_threads_atomic_double_reservation_budget',sorted(outcomes)==['blocked','reserved'] and a.charged_bytes()==268435456)
    good('fixture_threads_inherit_CPU26',os.sched_getaffinity(0)=={26})
    a,file,clock=instance(); attempt=a.reserve(1,None)
    empty={'bytes':0,'sha256':m.digest(b'')}
    a.state['attempts']=[dict(attempt,id=n,wire_name=a.filename(1,n,'wire'),canonical_name=a.filename(1,n,'canonical'),wire_meta=empty,canonical_meta=empty,wire_charged=0,canonical_charged=0,status='qualified') for n in range(296)]
    rejects('carry_in_4GET_cap_stops_at_300',lambda:a.reserve(2,None))
    a,file,clock=instance(); attempt=a.reserve(1,None); fetch(a,attempt); a.canonical_from_wire(attempt); a.qualify(attempt); clock.now+=2; a.record_calibration(); frozen=copy.deepcopy(a.state['calibration_cost']); a.save()
    clock.now=a.deadline-61
    rejects('same_deadline_late_cost_gate_refuses',a.continuation_gate)
    good('late_gate_does_not_rewrite_first_cost',a.state['calibration_cost']==frozen)
    index=a.canonical_index(); good('canonical_index_metadata_only_old_refs',index['pages'][0]['canonical_ref']['path']==prefix[0]['path'] and 'prices' not in json.dumps(index))
receipt={'status':'offline_artificial_transactions_only_not_live_admission','checks_passed':len(checks),'checks':checks,
         'CPU_affinity':sorted(os.sched_getaffinity(0)),'actual_HTTP_calls':0,'actual_credentials_accessed':False,
         'actual_market_rows_arrays_checkpoints_read':False,'Q3_read':False,
         'old_source_and_transport_globals_not_modified':True,
         'new_adapter_sha256':hashlib.sha256((BASE/'download_compressed_spy.py').read_bytes()).hexdigest(),
         'fixture_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(BASE/'OFFLINE_COMPRESSED_CHECKS.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ('status','checks_passed','new_adapter_sha256')}))
