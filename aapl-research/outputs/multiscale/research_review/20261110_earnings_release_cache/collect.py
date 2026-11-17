"""One bounded source-only cache window; completed documents survive interruption."""
import os
import signal
import subprocess
import sys
import time
from runtime import BASE,read,write,require,sha,setup,protocol

STOP=False
def stop(signum,frame):
    global STOP
    STOP=True

def main():
    setup();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
    require(not (BASE/'COLLECTION_PROTOCOL.json').exists(),'Do not open another window')
    review=read(BASE/'CODE_ACCEPTANCE.json');require(review['source_collection_accepted'],'Code review required')
    for f,h in review['reviewed_files'].items():require(sha(f)==h,'Reviewed file changed')
    sr=read(BASE/'SOURCE_ROSTER_AND_SCHEMA_REVIEW.json');require(sr['admission']['source_collection_only_admitted'],'Source/schema review required')
    spec=read(BASE/'COLLECTION_SPEC.json');events=read(BASE/'ROSTER.json')['events']
    require(len(events)==26 and sum('existing_body' not in e for e in events)==24,'Frozen26/24scope')
    require(24*(spec['response_cap_bytes']+1)<spec['aggregate_response_cap_bytes'],'Aggregate cap not implied by per-job cap')
    files=['runtime.py','fetch_worker.py','collect.py','ROSTER.json','SOURCE_SCHEMA.json','TWO_REPORT_SCHEMA_FIXTURE.json',
           'COLLECTION_SPEC.json','SOURCE_ROSTER_AND_SCHEMA_REVIEW.json','CODE_ACCEPTANCE.json']
    t=time.time();p={'started_epoch':t,'deadline_epoch':t+180,'locked_files':{str(BASE/f):sha(BASE/f) for f in files},
          'source_collection_only':True,'market_execution_admitted':False,'GPU':False,'CPU':[24]}
    write(BASE/'COLLECTION_PROTOCOL.json',p);output=BASE/'run';output.mkdir(exist_ok=False)
    checkpoint={'protocol_sha256':sha(BASE/'COLLECTION_PROTOCOL.json'),'completed':[],'active':[]};write(output/'CHECKPOINT.json',checkpoint)
    pending=[]
    for e in events:
        if 'existing_body' in e:
            c=e['existing_body'];require(sha(c['path'])==c['sha256'],'Existing source changed')
            folder=output/e['id'];folder.mkdir()
            write(folder/'RECEIPT.json',{'id':e['id'],'URL':e['URL'],'HTTP_attempts':0,'reused_body':c,
                  'complete_body':True,'HTTP_200_HTML':True,'content_qualification_pending':True,
                  'original_PIT_proved':False,'current_historical_version_assumed':True})
            checkpoint['completed'].append({'id':e['id'],'receipt_sha256':sha(folder/'RECEIPT.json')})
        else:pending.append(e)
    running={};launched=[]
    try:
        while pending or running:
            while pending and len(running)<2 and not STOP and time.time()<p['deadline_epoch']-15:
                e=pending.pop(0);stream=(output/(e['id']+'.log')).open('w')
                child=subprocess.Popen([sys.executable,'-B',str(BASE/'fetch_worker.py'),e['id']],stdout=stream,stderr=subprocess.STDOUT);stream.close()
                running[e['id']]={'process':child,'started':time.time()};launched.append(e['id'])
                require(len(launched)<=24,'Request cap exceeded')
                checkpoint['active']=[{'id':k,'pid':v['process'].pid} for k,v in running.items()];write(output/'CHECKPOINT.json',checkpoint)
            for id,job in list(running.items()):
                child=job['process']
                if child.poll() is None and time.time()-job['started']>13:
                    child.kill();child.wait(timeout=1)
                if child.poll() is not None:
                    file=output/id/'RECEIPT.json'
                    if not file.exists():write(file,{'id':id,'HTTP_attempts':1,'complete_body':False,'error':'worker terminal before receipt','pid':child.pid,'exit_code':child.returncode})
                    checkpoint['completed'].append({'id':id,'receipt_sha256':sha(file)})
                    del running[id]
                    checkpoint['active']=[{'id':k,'pid':v['process'].pid} for k,v in running.items()];write(output/'CHECKPOINT.json',checkpoint)
            if (STOP or time.time()>=p['deadline_epoch']-15) and not running:break
            time.sleep(.02)
        require(time.time()<p['deadline_epoch'],'Window expired')
        rows=[read(output/e['id']/'RECEIPT.json') for e in events if (output/e['id']/'RECEIPT.json').exists()]
        write(output/'COLLECTION_RESULT.json',{'protocol_sha256':sha(BASE/'COLLECTION_PROTOCOL.json'),'records':rows,
            'events_in_roster':26,'receipts':len(rows),'HTTP_attempts':sum(x['HTTP_attempts'] for x in rows),
            'complete_HTTP200_HTML_or_reused':sum(x.get('HTTP_200_HTML',False) and x['complete_body'] for x in rows),
            'all_jobs_terminal':not running,'unstarted_ids':[x['id'] for x in pending],'stopped_by_signal':STOP,
            'seconds':time.time()-t,'source_content_validation_pending':True,'market_execution_admitted':False,
            'GPU':False,'Q3_market_arrays_read':False,'new_model_fits':0})
    finally:
        for job in running.values():
            child=job['process']
            if child.poll() is None:child.kill();child.wait(timeout=1)

if __name__=='__main__':main()
