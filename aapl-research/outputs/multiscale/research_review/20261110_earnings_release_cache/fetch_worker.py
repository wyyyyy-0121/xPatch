"""One prelisted public issuer GET. No redirect, retry, market data or features."""
import os
import signal
import sys
import time
import urllib.request
import urllib.error
from runtime import BASE,read,write,require,sha,setup,protocol

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args):return None

def alarm(signum,frame):raise TimeoutError('12second absolute source request deadline')

def main():
    setup();p=protocol();spec=read(BASE/'COLLECTION_SPEC.json');roster=read(BASE/'ROSTER.json')['events']
    event=next(x for x in roster if x['id']==sys.argv[1]);require('existing_body' not in event,'No HTTP for existing cache')
    output=BASE/'run'/event['id'];output.mkdir(exist_ok=False)
    receipt={'id':event['id'],'URL':event['URL'],'pid':os.getpid(),'started_epoch':time.time(),
             'HTTP_attempts':1,'CPU_affinity':sorted(os.sched_getaffinity(0)),'source_only':True,
             'complete_body':False,'current_historical_version_assumed':True,'original_PIT_proved':False}
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,min(12,p['deadline_epoch']-time.time()))
    partial=output/'BODY.partial'
    try:
        require(time.time()<p['deadline_epoch']-12,'Insufficient unchanged window')
        req=urllib.request.Request(event['URL'],headers={'User-Agent':'Academic issuer-release source verification (single fixed event request)'})
        try:response=urllib.request.build_opener(NoRedirect).open(req,timeout=9)
        except urllib.error.HTTPError as exc:response=exc
        size=0
        with response:
            receipt.update(status=response.status,headers=dict(response.headers),content_type=response.headers.get('Content-Type',''))
            declared=response.headers.get('Content-Length')
            if declared is not None:
                require(declared.strip().isdigit(),'Invalid declared Content-Length')
                declared=int(declared.strip())
                require(declared<=spec['response_cap_bytes'],'Declared response exceeds byte cap')
            receipt['declared_content_length']=declared
            with partial.open('wb') as f:
                while True:
                    block=response.read(min(65536,spec['response_cap_bytes']+1-size))
                    if not block:break
                    size+=len(block);f.write(block)
                    require(size<=spec['response_cap_bytes'],'Response cap exceeded')
                f.flush();os.fsync(f.fileno())
        require(declared is None or size==declared,'Truncated or inconsistent declared body length')
        receipt.update(bytes=size,complete_body=True)
        receipt['completeness_basis']='exact_declared_length_and_EOF' if declared is not None else 'EOF_without_declared_length'
        os.replace(partial,output/'BODY.html')
        receipt['body_sha256']=sha(output/'BODY.html')
        receipt['HTTP_200_HTML']=receipt['status']==200 and 'html' in receipt['content_type'].lower()
        receipt['content_qualification_pending']=True
    except BaseException as exc:
        receipt.update(error=repr(exc),partial_bytes=partial.stat().st_size if partial.exists() else 0)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        receipt.update(finished_epoch=time.time(),elapsed_seconds=time.time()-receipt['started_epoch'])
        write(output/'RECEIPT.json',receipt)
    return 0

if __name__=='__main__':sys.exit(main())
