"""Independent bounded gzip continuation of fixed SPY history; no forecasts."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import threading
import time
import types
from urllib.parse import urlencode
from urllib.request import Request, build_opener

BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'20261110_market_context_coverage'
DEPENDENCIES={
    'source':{'path':str(OLD/'download_spy_coverage.py'),'sha256':'9cc0d1e2af800d4dfe33ac10db009d483f66f5e0912f2a4ca3f3953863c9b32a'},
    'transport':{'path':str(OLD/'transport_compression/probe_transport_gzip.py'),'sha256':'f7eaf73d8e5a0849761ec7839e91fad099d5cdaa007a055ceb9d5f9c677d6dcc'}}

def digest(data):
    return hashlib.sha256(data).hexdigest()


def locked_module(name):
    item=DEPENDENCIES[name]; body=Path(item['path']).read_bytes()
    if digest(body)!=item['sha256']:
        raise RuntimeError('dependency_identity_invalid')
    module=types.ModuleType('locked_compressed_'+name); module.__file__=item['path']
    exec(compile(body,item['path'],'exec'),module.__dict__)
    return module

SOURCE=locked_module('source'); TRANSPORT=locked_module('transport')
CAP=4*1024*1024; SIDE_RESERVE=CAP+1; ATTEMPT_RESERVE=2*SIDE_RESERVE
BUDGET={'window_seconds':3600,'global_GET':300,'prior_actual_GET':4,
        'payload_storage_bytes':256*1024*1024,'prior_payload_bytes':2127635,
        'wire_page_bytes':CAP,'canonical_page_bytes':CAP,'side_guard_bytes':1,
        'attempt_reserve_bytes':ATTEMPT_RESERVE,'workers':2,'CPU':24,'socket_seconds':15,'closing_reserve_seconds':2}
GATE={'kind':'SPY_range1_compressed_cost_gate_v1','calibration_range':1,'remaining_ranges':65,
      'factor':1.5,'closure_seconds':60,'inflight_safety_attempts':2,
      'inflight_safety_bytes':2*ATTEMPT_RESERVE,'requests_limit':300,
      'payload_storage_limit':268435456,'window_seconds':3600,'IO_workers':2,'IO_CPU':24,
      'calibration_bytes':'wire plus canonical actual persisted bytes',
      'serial_CPU_excludes_only_open_and_read':True}

class Halt(Exception):
    pass


def atomic(path,value):
    SOURCE.atomic(path,value)


def template(manifest_path=None):
    manifest_path=Path(manifest_path) if manifest_path else BASE/'INHERITANCE_MANIFEST.json'
    gate=BASE/'COST_GATE.json'
    if json.loads(gate.read_bytes())!=GATE:
        raise Halt('cost_gate_specification_invalid')
    return {'kind':'SPY_gzip_continuation_v1','execution_admitted':False,
            'source_sha256':digest(Path(__file__).read_bytes()),'dependencies':DEPENDENCIES,
            'inheritance_manifest_sha256':digest(manifest_path.read_bytes()),
            'cost_gate_sha256':digest(gate.read_bytes()),'budget':BUDGET,
            'endpoint':SOURCE.ENDPOINT,'fixed':SOURCE.FIXED,'ranges':SOURCE.RANGES,
            'Accept-Encoding':'gzip','started_epoch':None,'deadline_epoch':None,
            'GPU':False,'Q3_allowed':False,'features_fits_scores_allowed':False,
            'prior_window_closed_not_reset':True}


class Acquisition:
    def __init__(self,protocol_path,root=None,manifest_path=None,clock=time.time):
        self.root=Path(root) if root else BASE/'acquisition'
        self.root.mkdir(exist_ok=True,parents=True); self.path=self.root/'JOURNAL.json'
        self.manifest_path=Path(manifest_path) if manifest_path else BASE/'INHERITANCE_MANIFEST.json'
        protocol_bytes=Path(protocol_path).read_bytes(); self.protocol=json.loads(protocol_bytes)
        expected=template(self.manifest_path)
        for key,value in expected.items():
            if key not in ('execution_admitted','started_epoch','deadline_epoch') and self.protocol.get(key)!=value:
                raise Halt('protocol_identity_invalid')
        if self.protocol.get('execution_admitted') is not True:
            raise Halt('protocol_not_admitted')
        start,end=self.protocol.get('started_epoch'),self.protocol.get('deadline_epoch')
        if (type(start) not in (int,float) or type(end) not in (int,float)
                or not math.isfinite(start) or not math.isfinite(end) or end-start!=3600):
            raise Halt('new_absolute_window_invalid')
        self.protocol_hash=digest(protocol_bytes); self.clock=clock; self.deadline=end
        self.lock=threading.RLock(); self.stop=threading.Event(); self.sealed=False
        self.manifest=json.loads(self.manifest_path.read_bytes())
        if (self.manifest.get('kind')!='SPY_compressed_continuation_inheritance_v1'
                or self.manifest.get('prior_actual_GET')!=4 or self.manifest.get('prior_payload_bytes')!=2127635
                or self.manifest.get('old_windows_closed_not_reset') is not True):
            raise Halt('inheritance_contract_invalid')
        entities=self.manifest.get('entities',[])
        if (len(entities)!=4 or len({e['path'] for e in entities})!=4
                or sum(e['bytes'] for e in entities)!=2127635):
            raise Halt('inheritance_payload_accounting_invalid')
        self.prior_bytes=2127635
        if not self.path.exists() and any(p.name!='PROCESS.lock' for p in self.root.iterdir()):
            raise Halt('orphan_progress_without_journal_no_reset')
        self.state=json.loads(self.path.read_bytes()) if self.path.exists() else {
            'protocol_sha256':self.protocol_hash,'started_epoch':start,'deadline_epoch':end,
            'status':'prepared','attempts':[],'ranges':[{'pages':[],'complete':False} for _ in SOURCE.RANGES]}
        if (self.state.get('protocol_sha256')!=self.protocol_hash or self.state.get('started_epoch')!=start
                or self.state.get('deadline_epoch')!=end):
            raise Halt('journal_protocol_or_deadline_changed')
        attempts=self.state.get('attempts'); ranges=self.state.get('ranges')
        if not isinstance(attempts,list) or len(attempts)>296 or not isinstance(ranges,list) or len(ranges)!=67:
            raise Halt('journal_structure_invalid')
        for number,a in enumerate(attempts):
            if (a.get('id')!=number or type(a.get('range')) is not int or not 1<=a['range']<67
                    or type(a.get('page')) is not int or a['page']<0
                    or a.get('status') not in ('reserved','wire_committed','canonical_committed','qualified')):
                raise Halt('attempt_identity_invalid')
            if a.get('params')!=SOURCE.parameters(a['range'],a.get('params',{}).get('page_token')):
                raise Halt('attempt_parameters_invalid')
            for part in ('wire','canonical'):
                if a.get(part+'_name')!=self.filename(a['range'],number,part):
                    raise Halt('attempt_filename_invalid')
                meta=a.get(part+'_meta'); charge=a.get(part+'_charged')
                if meta is None:
                    if charge!=SIDE_RESERVE:
                        raise Halt('uncertain_reservation_invalid')
                elif (type(meta.get('bytes')) is not int or not 0<=meta['bytes']<=CAP
                        or charge!=meta['bytes'] or not re.fullmatch('[0-9a-f]{64}',str(meta.get('sha256')))):
                    raise Halt('committed_metadata_invalid')
            if ((a['status']=='reserved')!=(a.get('wire_meta') is None)
                    or (a['status'] in ('canonical_committed','qualified'))!=(a.get('canonical_meta') is not None)):
                raise Halt('attempt_transaction_state_invalid')
        if self.charged_bytes()>BUDGET['payload_storage_bytes']:
            raise Halt('inherited_and_new_storage_budget_invalid')
        self.guard(); self.audit_payload_inventory()
        self.state['sealed_snapshot']=False; self.save()

    @staticmethod
    def filename(index,number,part):
        return f'range_{index:03d}_attempt_{number:03d}.'+('wire' if part=='wire' else 'canonical.json')

    def artifact(self,a,part):
        return self.root/a[part+'_name']

    def charged_bytes(self):
        return self.prior_bytes+sum(a['wire_charged']+a['canonical_charged'] for a in self.state['attempts'])

    def audit_payload_inventory(self):
        allowed=set()
        for a in self.state['attempts']:
            for part in ('wire','canonical'):
                final=self.artifact(a,part); partial=final.with_suffix(final.suffix+'.partial')
                allowed.update((final.name,partial.name))
                if final.exists() and partial.exists():
                    raise Halt('duplicate_payload_copy_not_charged')
                for p in (final,partial):
                    if p.exists() and (not p.is_file() or p.stat().st_size>a[part+'_charged']):
                        raise Halt('payload_exceeds_existing_reservation')
        for p in self.root.iterdir():
            if ('.wire' in p.name or '.canonical.json' in p.name) and p.name not in allowed:
                raise Halt('orphan_payload_not_in_journal')

    def save(self):
        with self.lock:
            if self.sealed:
                raise Halt('journal_sealed')
            self.state.update(global_GET_charged=4+len(self.state['attempts']),
                              payload_storage_bytes_charged=self.charged_bytes())
            atomic(self.path,self.state)

    def guard(self):
        if (self.sealed or self.stop.is_set() or self.clock()<self.protocol['started_epoch']
                or self.clock()>=self.deadline-2):
            raise Halt('stopped_or_original_new_deadline')

    def seal(self,status):
        with self.lock:
            if self.sealed:
                return
            self.state.update(status=status,sealed_snapshot=True,features_fits_scores=False,
                              full_RTH_coverage_independently_proven=False,research_goal_achieved=False)
            self.save(); self.sealed=True

    def check_file(self,path,meta,limit=None):
        self.guard(); size=path.stat().st_size
        if size!=meta['bytes'] or (limit is not None and size>limit):
            raise Halt('artifact_size_invalid')
        h=hashlib.sha256()
        with path.open('rb') as stream:
            while True:
                self.guard(); block=stream.read(65536)
                if not block:
                    break
                h.update(block)
        if h.hexdigest()!=meta['sha256']:
            raise Halt('artifact_hash_invalid')

    def verify_inheritance(self):
        for e in self.manifest['evidence']:
            self.guard()
            if digest(Path(e['path']).read_bytes())!=e['sha256']:
                raise Halt('old_evidence_identity_changed')
        entities={e['path']:e for e in self.manifest['entities']}
        for e in entities.values():
            self.check_file(Path(e['path']),e)
        pages=sorted(self.manifest['prefix_pages'],key=lambda p:p['page'])
        if len(pages)!=2:
            raise Halt('inherited_prefix_page_count_invalid')
        previous=None; token=None; qualified=[]
        for number,p in enumerate(pages):
            if (p['page']!=number or p['params']!=SOURCE.parameters(0,token)
                    or p['path'] not in entities or p['bytes']!=entities[p['path']]['bytes']
                    or p['sha256']!=entities[p['path']]['sha256'] or (number and token is None)):
                raise Halt('inherited_prefix_chain_invalid')
            body=Path(p['path']).read_bytes()
            meta=SOURCE.validate_page(body,0,previous,self.guard)
            qualified.append(self.index_page(0,number,p,meta,token))
            previous=meta['last_UTC']; token=meta['next_token']
        if token is not None:
            raise Halt('inherited_prefix_not_terminal')
        self.state['ranges'][0]={'pages':qualified,'complete':True,'inherited':True}

    @staticmethod
    def index_page(index,page,ref,meta,request_token):
        return {'range':index,'page':page,'canonical_ref':{k:ref[k] for k in ('path','bytes','sha256')},
                'request_page_token':request_token,'next_page_token':meta['next_token'],
                'rows':meta['rows'],'first_UTC':meta['first_UTC'],'last_UTC':meta['last_UTC']}

    def reserve(self,index,token):
        with self.lock:
            self.guard()
            if self.deadline-self.clock()<=17:
                raise Halt('insufficient_time_for_new_GET')
            pages=self.state['ranges'][index]['pages']
            expected=pages[-1]['next_page_token'] if pages else None
            if self.state['ranges'][index]['complete'] or token!=expected or index==0:
                raise Halt('range_terminal_or_request_token_invalid')
            attempts=self.state['attempts']; self.audit_payload_inventory()
            if 4+len(attempts)>=300 or self.charged_bytes()+ATTEMPT_RESERVE>268435456:
                raise Halt('cumulative_GET_or_storage_budget_exhausted')
            number=len(attempts)
            a={'id':number,'range':index,'page':len(pages),'params':SOURCE.parameters(index,token),
               'status':'reserved','wire_charged':SIDE_RESERVE,'canonical_charged':SIDE_RESERVE,
               'wire_meta':None,'canonical_meta':None,
               'wire_name':self.filename(index,number,'wire'),
               'canonical_name':self.filename(index,number,'canonical')}
            attempts.append(a); self.save(); return a

    def commit(self,a,part,meta):
        with self.lock:
            self.guard()
            final=self.artifact(a,part); partial=final.with_suffix(final.suffix+'.partial')
            if partial.stat().st_size!=meta['bytes'] or meta['bytes']>CAP:
                raise Halt('complete_temp_size_invalid')
            old=a.get(part+'_meta')
            if old is not None and old!=meta:
                raise Halt('reconstructed_committed_identity_changed')
            a[part+'_meta']=meta; a[part+'_charged']=meta['bytes']
            a['status']='wire_committed' if part=='wire' else 'canonical_committed'
            self.save(); self.guard(); partial.replace(final); self.save()

    def fetch_wire(self,a,auth):
        request=Request(SOURCE.ENDPOINT+'?'+urlencode(a['params']),headers={
            'APCA-API-KEY-ID':auth[0],'APCA-API-SECRET-KEY':auth[1],
            'User-Agent':'AAPL-SPY-source-coverage/1.0','Accept-Encoding':'gzip'})
        partial=self.artifact(a,'wire').with_suffix('.wire.partial')
        io_wall=0.0; h=hashlib.sha256(); count=0
        opener=build_opener(SOURCE.RejectRedirect)
        try:
            self.guard(); stamp=time.monotonic()
            try:
                response=opener.open(request,timeout=15)
            finally:
                io_wall+=time.monotonic()-stamp
            with response:
                if response.status!=200 or response.url.split('?')[0]!=SOURCE.ENDPOINT:
                    raise Halt('response_endpoint_or_status_invalid')
                values=response.headers.get_all('Content-Encoding',[]) if hasattr(response.headers,'get_all') else [response.headers.get('Content-Encoding')]
                if len(values)!=1 or str(values[0]).strip().lower()!='gzip':
                    raise Halt('gzip_required_no_identity_fallback')
                headers={k.lower():v for k,v in response.headers.items()
                         if k.lower() in ('date','content-type','content-encoding','content-length','x-request-id')}
                with partial.open('xb') as stream:
                    while True:
                        self.guard(); stamp=time.monotonic()
                        try:
                            block=response.read(min(65536,SIDE_RESERVE-count))
                        finally:
                            io_wall+=time.monotonic()-stamp
                        if not block:
                            stream.flush(); os.fsync(stream.fileno()); break
                        stream.write(block); h.update(block); count+=len(block)
                        if count>CAP:
                            raise Halt('wire_cap_exceeded')
            self.commit(a,'wire',{'bytes':count,'sha256':h.hexdigest(),'headers':headers})
        finally:
            with self.lock:
                if not self.sealed:
                    a['io_wall_seconds']=io_wall; self.save()

    def canonical_from_wire(self,a):
        wire=self.artifact(a,'wire'); self.check_file(wire,a['wire_meta'],CAP)
        canonical=self.artifact(a,'canonical'); partial=canonical.with_suffix(canonical.suffix+'.partial')
        if partial.exists() and partial.stat().st_size>CAP:
            raise Halt('canonical_partial_cap_invalid')
        original_size=partial.stat().st_size if partial.exists() else 0
        owner=self
        with partial.open('r+b' if partial.exists() else 'x+b') as output:
            class SavingDecoder(TRANSPORT.Decoder):
                def output(self,piece):
                    owner.guard(); super().output(piece)
                    position=output.tell(); overlap=min(len(piece),max(0,original_size-position))
                    if overlap and output.read(overlap)!=piece[:overlap]:
                        raise Halt('canonical_partial_prefix_mismatch_preserved')
                    if overlap<len(piece):
                        output.write(piece[overlap:])
            decoder=SavingDecoder('gzip')
            with wire.open('rb') as stream:
                while True:
                    self.guard(); block=stream.read(65536)
                    if not block:
                        break
                    decoder.feed(block)
            meta=decoder.finish()
            if (meta['encoded_bytes']!=a['wire_meta']['bytes'] or meta['encoded_sha256']!=a['wire_meta']['sha256']
                    or output.tell()<original_size):
                raise Halt('wire_decode_or_existing_partial_identity_invalid')
            output.flush(); os.fsync(output.fileno())
        self.commit(a,'canonical',{'bytes':meta['decoded_bytes'],'sha256':meta['decoded_sha256']})

    def qualify(self,a):
        with self.lock:
            current=self.state['ranges'][a['range']]; pages=current['pages']
            if current['complete'] or a['page']!=len(pages):
                raise Halt('page_gap_or_page_after_terminal')
            prior=pages[-1] if pages else None; token=prior['next_page_token'] if prior else None
            if a['params']!=SOURCE.parameters(a['range'],token):
                raise Halt('canonical_request_token_chain_invalid')
        canonical=self.artifact(a,'canonical'); self.check_file(canonical,a['canonical_meta'],CAP)
        meta=SOURCE.validate_page(canonical.read_bytes(),a['range'],prior['last_UTC'] if prior else None,self.guard)
        if meta['next_token'] is not None and meta['next_token'] in [p['next_page_token'] for p in pages]:
            raise Halt('pagination_cycle')
        ref={'path':str(canonical),'bytes':a['canonical_meta']['bytes'],'sha256':a['canonical_meta']['sha256']}
        with self.lock:
            self.guard(); current['pages'].append(self.index_page(a['range'],a['page'],ref,meta,token))
            current['complete']=meta['next_token'] is None; a['status']='qualified'; self.save()

    def promote(self,a,part):
        final=self.artifact(a,part); temp=final.with_suffix(final.suffix+'.partial'); meta=a[part+'_meta']
        if final.is_file():
            self.check_file(final,meta,CAP); return True
        if temp.is_file() and temp.stat().st_size==meta['bytes']:
            self.check_file(temp,meta,CAP)
            with self.lock:
                self.guard(); temp.replace(final)
            return True
        return False

    def recover(self):
        self.audit_payload_inventory(); self.verify_inheritance()
        for current in self.state['ranges'][1:]:
            current.update(pages=[],complete=False)
        uncertain=False
        for a in sorted(self.state['attempts'],key=lambda a:(a['range'],a['page'])):
            self.guard()
            if a['wire_meta'] is None:
                uncertain=True; continue
            if not self.promote(a,'wire'):
                raise Halt('committed_wire_missing')
            if a['canonical_meta'] is None or not self.promote(a,'canonical'):
                self.canonical_from_wire(a)
            self.qualify(a)
        if uncertain:
            raise Halt('unknown_old_attempt_preserved_no_retry')
        self.save()

    def record_calibration(self):
        if 'calibration_cost' in self.state:
            return
        attempts=[a for a in self.state['attempts'] if a['range']==1]
        if not attempts or not self.state['ranges'][1]['complete'] or any(a['status']!='qualified' for a in attempts):
            raise Halt('calibration_incomplete')
        end=self.clock()
        self.state['calibration_completed_epoch']=end
        self.state['calibration_cost']={'p1':len(attempts),
            'b1':sum(a['wire_charged']+a['canonical_charged'] for a in attempts),
            't1_seconds':end-self.protocol['started_epoch'],
            'io_wall_seconds':sum(a['io_wall_seconds'] for a in attempts)}

    def continuation_gate(self):
        attempts=[a for a in self.state['attempts'] if a['range']==1]
        c=self.state.get('calibration_cost',{}); end=self.state.get('calibration_completed_epoch')
        if (not attempts or not self.state['ranges'][1]['complete'] or any(a['status']!='qualified' for a in attempts)
                or c.get('p1')!=len(attempts)
                or c.get('b1')!=sum(a['wire_charged']+a['canonical_charged'] for a in attempts)
                or c.get('io_wall_seconds')!=sum(a['io_wall_seconds'] for a in attempts)
                or type(end) not in (int,float) or c.get('t1_seconds')!=end-self.protocol['started_epoch']):
            raise Halt('frozen_calibration_cost_changed')
        io=c['io_wall_seconds']; total=c['t1_seconds']
        if not math.isfinite(io) or not math.isfinite(total) or io<0 or total<=0 or total>3600:
            raise Halt('calibration_time_invalid')
        cpu=max(0,total-io); req=math.ceil(65*c['p1']*1.5); size=math.ceil(65*c['b1']*1.5)
        wall=65*1.5*(io/2+cpu)
        accepted=(4+len(self.state['attempts'])+req<=300
                  and self.charged_bytes()+size+2*ATTEMPT_RESERVE<=268435456
                  and self.clock()+wall+60<=self.deadline)
        self.state['continuation_gate']={'accepted':accepted,'cost_gate_sha256':self.protocol['cost_gate_sha256'],
            'remaining_GET_predicted':req,'remaining_storage_bytes_predicted':size,
            'remaining_wall_seconds_predicted':wall,'calibration_cost':dict(c),'serial_seconds':cpu,
            'evaluated_epoch':self.clock(),'not_completion_guarantee':True}
        self.save()
        if not accepted:
            raise Halt('fixed_compressed_cost_gate_failed')

    def canonical_index(self):
        return {'kind':'SPY_qualified_canonical_index_v1','protocol_sha256':self.protocol_hash,
                'inheritance_manifest_sha256':self.protocol['inheritance_manifest_sha256'],
                'all_67_ranges_complete':all(r['complete'] for r in self.state['ranges']),
                'ranges':SOURCE.RANGES,'pages':[p for r in self.state['ranges'] for p in r['pages']],
                'source_only_not_RTH_or_context_admission':True,'Q3_allowed':False}

    def worker(self,tasks,auth):
        os.sched_setaffinity(0,{24})
        try:
            while not self.stop.is_set():
                try:
                    index=tasks.get_nowait()
                except queue.Empty:
                    return
                while not self.state['ranges'][index]['complete']:
                    pages=self.state['ranges'][index]['pages']; token=pages[-1]['next_page_token'] if pages else None
                    a=self.reserve(index,token); self.fetch_wire(a,auth); self.canonical_from_wire(a); self.qualify(a)
        except Exception as exc:
            with self.lock:
                if not self.sealed:
                    self.state.update(error_type=type(exc).__name__,reason_code=str(exc) if isinstance(exc,(Halt,SOURCE.Halt,TRANSPORT.Halt)) else 'source_or_runtime_failure')
                    self.stop.set(); self.save()

    def run(self,scope):
        self.guard(); self.recover()
        if scope=='remaining':
            self.continuation_gate()
        indices=[1] if scope=='calibration' else list(range(2,67))
        tasks=queue.Queue()
        for index in indices:
            if not self.state['ranges'][index]['complete']:
                tasks.put(index)
        if tasks.empty():
            if scope=='calibration':
                self.record_calibration()
            with self.lock:
                if all(r['complete'] for r in self.state['ranges']):
                    atomic(self.root/'CANONICAL_INDEX.json',self.canonical_index())
                self.seal('requested_scope_already_complete')
            return
        auth=SOURCE.credentials(); self.state.update(status='running',scope=scope); self.save()
        threads=[threading.Thread(target=self.worker,args=(tasks,auth),daemon=True)
                 for _ in range(min(2,tasks.qsize()))]
        for thread in threads:
            thread.start()
        while any(t.is_alive() for t in threads) and not self.stop.wait(0.05):
            if self.clock()>=self.deadline-2:
                self.stop.set()
        with self.lock:
            if self.stop.is_set():
                status='stopped_preserve_progress'
            elif scope=='calibration':
                self.record_calibration(); status='calibration_complete_awaiting_root_gate'
            else:
                status='all_ranges_downloaded_field_validated'
                atomic(self.root/'CANONICAL_INDEX.json',self.canonical_index())
            self.seal(status)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--template',action='store_true')
    parser.add_argument('--protocol',type=Path); parser.add_argument('--scope',choices=('calibration','remaining'))
    args=parser.parse_args()
    if args.template:
        print(json.dumps(template(),indent=2)); return
    if not args.protocol or not args.scope:
        parser.error('Need --protocol and --scope; no request sent')
    os.sched_setaffinity(0,{24}); root=BASE/'acquisition'; root.mkdir(exist_ok=True)
    with (root/'PROCESS.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB); acquisition=None
        try:
            acquisition=Acquisition(args.protocol); acquisition.guard()
            def stop(_sig,_frame):
                acquisition.stop.set()
            for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGUSR1):
                signal.signal(sig,stop)
            signal.setitimer(signal.ITIMER_REAL,acquisition.deadline-2-time.time())
            acquisition.run(args.scope)
        except Exception as exc:
            if acquisition is not None:
                with acquisition.lock:
                    acquisition.state.update(error_type=type(exc).__name__,reason_code=str(exc) if isinstance(exc,(Halt,SOURCE.Halt,TRANSPORT.Halt)) else 'source_or_runtime_failure')
                    acquisition.seal('stopped_preserve_progress')
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
        result={'status':'not_started'} if acquisition is None else {
            k:acquisition.state.get(k) for k in ('status','global_GET_charged','payload_storage_bytes_charged','reason_code')}
        print(json.dumps(result))
        if result['status'] not in ('calibration_complete_awaiting_root_gate','all_ranges_downloaded_field_validated','requested_scope_already_complete'):
            raise SystemExit(75)

if __name__=='__main__':
    main()
