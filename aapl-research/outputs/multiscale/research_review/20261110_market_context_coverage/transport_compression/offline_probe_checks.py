"""Artificial-data, fake-opener checks only. Never calls actual HTTP/auth."""
from pathlib import Path
import email.message
import gzip
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import time
import zlib

BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('probe_under_test',BASE/'probe_transport_gzip.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
assert os.sched_getaffinity(0)=={26}
checks=[]
def good(name, value):
    assert value,name
    checks.append({'name':name,'passed':True})
def rejects(name, call):
    try:
        call()
    except (m.Halt,zlib.error):
        good(name,True)
    else:
        raise AssertionError(name)
def decode(body,encoding='gzip'):
    d=m.Decoder(encoding)
    for offset in range(0,len(body),65536):
        d.feed(body[offset:offset+65536])
    return d.finish()

realref=m.reference()
good('only_accepted_reference_metadata_hash',m.REFERENCE_SHA==hashlib.sha256((BASE/'REFERENCE_FIRST_REQUEST.json').read_bytes()).hexdigest())
plain=json.dumps({'artificial_constant_fixture':'x'*10000},separators=(',',':')).encode()
packed=gzip.compress(plain,mtime=0)
good('gzip_decoded_hash_equals_plain',decode(packed)['decoded_sha256']==hashlib.sha256(plain).hexdigest())
good('identity_missing_encoding',decode(plain,None)['decoded_bytes']==len(plain))
for coding in ('br','deflate','gzip,identity','', 'x-gzip'):
    rejects('unsupported_encoding_'+repr(coding),lambda coding=coding:decode(packed,coding))
for name,data in [('truncated',packed[:-1]),('second_member',packed+packed),('trailing_zero',packed+b'\0')]:
    rejects(name,lambda data=data:decode(data))
bad=bytearray(packed); bad[-8]^=1
rejects('CRC_failure',lambda:decode(bytes(bad)))
good('exact_decoded_4MiB',decode(gzip.compress(b'x'*m.CAP,mtime=0))['decoded_bytes']==m.CAP)
rejects('decoded_cap_plus_one',lambda:decode(gzip.compress(b'x'*(m.CAP+1),mtime=0)))
rejects('zip_bomb_decoded_guard',lambda:decode(gzip.compress(b'x'*(4*m.CAP),mtime=0)))
rejects('encoded_cap_plus_one',lambda:decode(b'x'*(m.CAP+1),'identity'))
now=time.time(); protocol=dict(m.template(),execution_admitted=True,started_epoch=now,deadline_epoch=now+120)
class Response(io.BytesIO):
    def __init__(self,body,encoding):
        super().__init__(body); self.status=200; self.url=m.ENDPOINT
        self.headers=email.message.Message()
        if encoding is not None:
            self.headers['Content-Encoding']=encoding
class Opener:
    def __init__(self,body,encoding):
        self.calls=0; self.body=body; self.encoding=encoding; self.request=None
    def open(self,request,timeout):
        self.calls+=1; self.request=request
        assert timeout==15
        return Response(self.body,self.encoding)
with tempfile.TemporaryDirectory(prefix='synthetic_gzip_transport_') as directory:
    p=Path(directory)/'protocol.json'; p.write_text(json.dumps(protocol)); m.validate_protocol(p)
    good('admitted_protocol_validates',True)
    invalid=dict(protocol,execution_admitted=False); p.write_text(json.dumps(invalid))
    rejects('unadmitted_protocol_rejected',lambda:m.validate_protocol(p))
    invalid=dict(protocol,deadline_epoch=protocol['deadline_epoch']+1); p.write_text(json.dumps(invalid))
    rejects('deadline_cannot_extend',lambda:m.validate_protocol(p))
    rejects('expired_window_rejected',lambda:m.guard(dict(protocol,started_epoch=now-121,deadline_epoch=now-1)))
    m.reference=lambda:dict(realref,body_bytes=len(plain),body_sha256=hashlib.sha256(plain).hexdigest())
    for name,body,coding in [('gzip',packed,'gzip'),('identity',plain,'identity'),('mismatch',gzip.compress(b'other',mtime=0),'gzip')]:
        run=Path(directory)/name; run.mkdir(); m.BASE=run
        opener=Opener(body,coding)
        result=m.execute(protocol,'artificial_protocol_hash',('synthetic-key','synthetic-secret'),opener)
        good(name+'_single_GET_only',opener.calls==1)
        good(name+'_wire_saved_hash',hashlib.sha256((run/'probe_run/response.wire').read_bytes()).hexdigest()==result['encoded_sha256'])
        good(name+'_journal_sealed',result['sealed_snapshot'] is True)
        if name=='gzip':
            good('gzip_passes_only_exact_hash_and_byte_reduction',result['status']=='gzip_transport_probe_passed' and result['decoded_identity_matches_accepted_first_page'])
            good('only_extra_header_is_gzip',opener.request.get_header('Accept-encoding')=='gzip')
            rejects('repeat_invocation_no_new_GET',lambda:m.execute(protocol,'artificial_protocol_hash',('synthetic-key','synthetic-secret'),opener))
            good('repeat_invocation_still_one_GET',opener.calls==1)
        elif name=='identity':
            good('identity_not_success_no_second_request',result['reason_code']=='gzip_not_supported_identity_response_no_retry')
        else:
            good('mismatch_stops_no_semantic_fallback',result['reason_code']=='decoded_identity_mismatch_no_semantic_fallback')
receipt={'status':'offline_fake_opener_synthetic_only_not_live_admission','checks_passed':len(checks),'checks':checks,
         'CPU_affinity':sorted(os.sched_getaffinity(0)),'actual_HTTP_calls':0,'actual_credentials_accessed':False,
         'actual_market_raw_arrays_checkpoints_read':False,'accepted_JOURNAL_metadata_only_reference':True,
         'adapter_sha256':hashlib.sha256((BASE/'probe_transport_gzip.py').read_bytes()).hexdigest(),
         'fixture_runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(BASE/'OFFLINE_PROBE_CHECKS.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({key:receipt[key] for key in ('status','checks_passed','adapter_sha256')}))
