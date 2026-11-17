"""One admitted GET of existing SPY page, gzip transport only; no forecasting."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import time
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zlib

BASE = Path(__file__).resolve().parent
ENDPOINT = 'https://data.alpaca.markets/v2/stocks/bars'
REFERENCE_SHA = '39157b49daf34d38270627476a44bcf718d81a225431e3a68c330a2cc34e7d5c'
CAP = 4 * 1024 * 1024
BUDGET = {'window_seconds': 120, 'requests': 1, 'socket_seconds': 15,
          'encoded_page_bytes': CAP, 'decoded_page_bytes': CAP,
          'encoded_guard_bytes': 1, 'decoded_guard_bytes': 1,
          'CPU': 24, 'workers': 1, 'closing_reserve_seconds': 2}

class Halt(Exception):
    pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def reference():
    data=(BASE/'REFERENCE_FIRST_REQUEST.json').read_bytes()
    if sha(data) != REFERENCE_SHA:
        raise Halt('reference_identity_invalid')
    return json.loads(data)


def template():
    reference()
    return {'kind':'SPY_single_page_gzip_transport_probe_v1', 'execution_admitted':False,
            'source_sha256':sha(Path(__file__).read_bytes()), 'reference_sha256':REFERENCE_SHA,
            'endpoint':ENDPOINT, 'Accept-Encoding':'gzip', 'budget':BUDGET,
            'started_epoch':None, 'deadline_epoch':None,
            'GPU':False,'Q3_allowed':False,'features_fits_scores_allowed':False,
            'full_history_admitted':False,'automatic_continuation_allowed':False}


def credentials():
    names=('APCA_API_KEY_ID','APCA_API_SECRET_KEY')
    values={name:os.environ.get(name,'') for name in names}
    path=Path.home()/'.config/alpaca/credentials.env'
    if not all(values.values()) and path.is_file():
        for raw in path.read_text().splitlines():
            line=raw.strip()
            if line.startswith('export '):
                line=line[7:].strip()
            name,sep,value=line.partition('='); name=name.strip()
            if sep and name in names and not values[name]:
                parsed=shlex.split(value,comments=True)
                if len(parsed)!=1:
                    raise Halt('credential_literal_invalid')
                values[name]=parsed[0]
    if not all(values.values()):
        raise Halt('credentials_unavailable_no_request')
    return tuple(values[name] for name in names)


class RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Decoder:
    """Streaming bounded one-member decoder. Does not retain market values."""
    def __init__(self, encoding):
        self.encoding='identity' if encoding is None else encoding.strip().lower()
        if self.encoding not in ('gzip','identity'):
            raise Halt('unsupported_or_stacked_content_encoding')
        self.decoder=zlib.decompressobj(16+zlib.MAX_WBITS) if self.encoding=='gzip' else None
        self.encoded_bytes=self.decoded_bytes=0
        self.encoded_hash=hashlib.sha256(); self.decoded_hash=hashlib.sha256()

    def output(self, piece):
        self.decoded_bytes+=len(piece)
        self.decoded_hash.update(piece)
        if self.decoded_bytes>CAP:
            raise Halt('decoded_cap_exceeded')

    def feed(self, block):
        self.encoded_bytes+=len(block); self.encoded_hash.update(block)
        if self.encoded_bytes>CAP:
            raise Halt('encoded_cap_exceeded')
        if self.decoder is None:
            self.output(block); return
        pending=block
        if self.decoder.eof and pending:
            raise Halt('trailing_or_second_gzip_member')
        while pending:
            before=len(pending)
            piece=self.decoder.decompress(pending,min(65536,CAP-self.decoded_bytes+1))
            self.output(piece); pending=self.decoder.unconsumed_tail
            if self.decoder.eof:
                if self.decoder.unused_data or pending:
                    raise Halt('trailing_or_second_gzip_member')
                break
            if not piece and len(pending)==before:
                raise Halt('decoder_no_progress')

    def finish(self):
        if self.decoder is not None and not self.decoder.eof:
            raise Halt('truncated_gzip')
        return {'content_encoding':self.encoding,'encoded_bytes':self.encoded_bytes,
                'encoded_sha256':self.encoded_hash.hexdigest(),
                'decoded_bytes':self.decoded_bytes,'decoded_sha256':self.decoded_hash.hexdigest(),
                'gzip_EOF_CRC_verified':self.decoder is not None}


def validate_protocol(path):
    data=path.read_bytes(); protocol=json.loads(data); expected=template()
    for key,value in expected.items():
        if key not in ('execution_admitted','started_epoch','deadline_epoch') and protocol.get(key)!=value:
            raise Halt('protocol_identity_invalid')
    if protocol.get('execution_admitted') is not True:
        raise Halt('protocol_not_admitted')
    start,end=protocol.get('started_epoch'),protocol.get('deadline_epoch')
    if (type(start) not in (int,float) or type(end) not in (int,float)
            or not math.isfinite(start) or not math.isfinite(end)
            or end-start!=BUDGET['window_seconds']):
        raise Halt('original_120_second_window_invalid')
    return protocol,sha(data)


def guard(protocol):
    now=time.time()
    if now<protocol['started_epoch'] or now>=protocol['deadline_epoch']-BUDGET['closing_reserve_seconds']:
        raise Halt('outside_original_window_no_reset')


def execute(protocol, protocol_hash, auth, opener=None):
    root=BASE/'probe_run'; root.mkdir(exist_ok=True)
    path=root/'JOURNAL.json'
    if path.exists():
        raise Halt('single_GET_journal_exists_no_retry')
    ref=reference(); guard(protocol)
    state={'protocol_sha256':protocol_hash,'reference_sha256':REFERENCE_SHA,
           'started_epoch':protocol['started_epoch'],'deadline_epoch':protocol['deadline_epoch'],
           'requests_charged':1,'status':'reserved_before_single_GET',
           'encoded_bytes_reserved':CAP+1,'decoded_bytes_reserved':CAP+1,
           'params':ref['params'],'request_Accept-Encoding':'gzip',
           'features_fits_scores':False,'full_history_admitted':False,'research_goal_achieved':False}
    atomic(path,state)
    wire=root/'response.wire'; partial=root/'response.wire.partial'
    decoder=None; complete=False; io_wall=0.0
    wire_saved_hash=hashlib.sha256(); wire_saved_bytes=0
    try:
        guard(protocol)
        request=Request(ENDPOINT+'?'+urlencode(ref['params']),headers={
            'APCA-API-KEY-ID':auth[0],'APCA-API-SECRET-KEY':auth[1],
            'User-Agent':'AAPL-SPY-source-coverage/1.0','Accept-Encoding':'gzip'})
        opener=build_opener(RejectRedirect) if opener is None else opener
        stamp=time.monotonic()
        try:
            response=opener.open(request,timeout=BUDGET['socket_seconds'])
        finally:
            io_wall+=time.monotonic()-stamp
        with response:
            if response.status!=200 or response.url.split('?')[0]!=ENDPOINT:
                raise Halt('response_endpoint_or_status_invalid')
            encoding_values=response.headers.get_all('Content-Encoding',[]) if hasattr(response.headers,'get_all') else [response.headers.get('Content-Encoding')]
            if len(encoding_values)>1:
                raise Halt('multiple_content_encoding_headers')
            encoding=encoding_values[0] if encoding_values else None
            decoder=Decoder(encoding)
            state['headers']={k.lower():v for k,v in response.headers.items()
                              if k.lower() in ('date','content-type','content-encoding','content-length','x-request-id')}
            with partial.open('xb') as stream:
                while True:
                    guard(protocol); stamp=time.monotonic()
                    try:
                        block=response.read(min(65536,CAP+1-decoder.encoded_bytes))
                    finally:
                        io_wall+=time.monotonic()-stamp
                    if not block:
                        stream.flush(); os.fsync(stream.fileno()); break
                    stream.write(block); wire_saved_hash.update(block); wire_saved_bytes+=len(block)
                    decoder.feed(block); guard(protocol)
                meta=decoder.finish(); guard(protocol)
            state.update(meta,wire_complete=True,status='encoded_body_committed_pending_equivalence')
            atomic(path,state); partial.replace(wire)
            complete=True
        equal=(meta['decoded_bytes']==ref['body_bytes'] and meta['decoded_sha256']==ref['body_sha256'])
        state['decoded_identity_matches_accepted_first_page']=equal
        if decoder.encoding=='identity':
            raise Halt('gzip_not_supported_identity_response_no_retry')
        if not equal:
            raise Halt('decoded_identity_mismatch_no_semantic_fallback')
        if meta['encoded_bytes']>=meta['decoded_bytes']:
            raise Halt('gzip_no_byte_reduction')
        state.update(status='gzip_transport_probe_passed',source_field_identity_preserved=True,
                     source_field_qualification_by_exact_accepted_page_hash=True)
    except Exception as exc:
        state.update(status='stopped_preserve_progress',error_type=type(exc).__name__,
                     reason_code=str(exc) if isinstance(exc,Halt) else 'transport_integrity_or_runtime_failure')
    finally:
        state.update(io_wall_seconds=io_wall,wire_complete=complete,sealed_snapshot=True,
                     wire_saved_bytes=wire_saved_bytes,wire_saved_sha256=wire_saved_hash.hexdigest())
        if decoder is not None:
            state.update(encoded_bytes_observed=decoder.encoded_bytes,
                         decoded_bytes_observed=decoder.decoded_bytes)
        atomic(path,state)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template',action='store_true'); parser.add_argument('--protocol',type=Path)
    args=parser.parse_args()
    if args.template:
        print(json.dumps(template(),indent=2)); return
    if not args.protocol:
        parser.error('Need --protocol; no request sent')
    timer=False
    try:
        protocol,protocol_hash=validate_protocol(args.protocol); guard(protocol)
        os.sched_setaffinity(0,{BUDGET['CPU']})
        def stop(_sig,_frame):
            raise Halt('signal_or_original_deadline')
        for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGUSR1):
            signal.signal(sig,stop)
        signal.setitimer(signal.ITIMER_REAL,protocol['deadline_epoch']-BUDGET['closing_reserve_seconds']-time.time()); timer=True
        root=BASE/'probe_run'; root.mkdir(exist_ok=True)
        with (root/'PROCESS.lock').open('a') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            auth=credentials(); state=execute(protocol,protocol_hash,auth)
        print(json.dumps({k:state.get(k) for k in ('status','requests_charged','reason_code','encoded_bytes','decoded_bytes','decoded_identity_matches_accepted_first_page')}))
        code=0 if state['status']=='gzip_transport_probe_passed' else 75
    except Exception as exc:
        print(json.dumps({'status':'not_started_or_existing_progress_preserved','error_type':type(exc).__name__,
                          'reason_code':str(exc) if isinstance(exc,Halt) else 'runtime_preflight_failure'})); code=75
    finally:
        if timer:
            signal.setitimer(signal.ITIMER_REAL,0)
    raise SystemExit(code)

if __name__=='__main__':
    main()
