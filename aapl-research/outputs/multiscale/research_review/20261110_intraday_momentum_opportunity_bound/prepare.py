"""Freeze metadata only, without importing numerical libraries or decoding arrays."""
import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from runtime import BASE, require, read, write, sha

def main():
    repo=BASE.parents[3]
    previous=read(BASE.parent/'20261110_level_constraint_qualification/DESIGN_SPEC.json')
    old=BASE.parent/'20261110_function_centering_matched_probe'
    metadata=read(old/'FUTURE_INPUT_IDENTITIES_METADATA_ONLY.json')
    admission=read(repo/'outputs/multiscale/admission/alpaca_20261107T052228573242Z/admission.json')
    bars={}
    for name,rows in [('minute',535620),('hourly',8238)]:
        item=dict(metadata['bars'][name]); item['rows']=rows
        require(admission['files'][__import__('pathlib').Path(item['path']).name]==item['sha256'],'Bar metadata differs')
        bars[name]=item
    modules={distribution:{'module':module,'file':str(Path(importlib.util.find_spec(module).origin).resolve())}
             for distribution,module in [('numpy','numpy'),('pandas','pandas'),('exchange-calendars','exchange_calendars'),('tzdata','tzdata')]}
    calendar_root=Path(modules['exchange-calendars']['file']).parent
    calendar_sources={str(calendar_root/name):sha(calendar_root/name) for name in
           ['calendar_utils.py','exchange_calendar.py','exchange_calendar_xnys.py','us_holidays.py']}
    spec={'schema':'intraday_opportunity_bound_design_v1','created_before_new_array_decode':True,
          'states':previous['states'],'bars':bars,
          'calendar':{'name':'XNYS','start':'2021-01-04','end':'2026-06-30',
                      'policy':'historical scheduled open/close, not point-in-time calendar receipt'},
          'package_versions':{k:importlib.metadata.version(k) for k in ['numpy','pandas','exchange-calendars','tzdata']},
          'modules':modules,'calendar_source_files':calendar_sources,
          'python':{'version':sys.version,'executable':str(Path(sys.executable).resolve())},
          'prediction_archive_members':previous['permitted_members'],
          'prediction_decode_members':['origins','labels','zero','prefix_update_mean','same_information_Ridge'],
          'bar_archive_members':['start_ns','end_ns','close','observed','session_date'],
          'bar_decode_members':['start_ns','end_ns','observed','session_date'],
          'source_files':previous['source_files']|{str(repo/'multiscale/bars.py'):sha(repo/'multiscale/bars.py'),
               str(repo/'aapl/data.py'):sha(repo/'aapl/data.py')},
          'source_metadata':previous['source_metadata']|{str(old/'FUTURE_INPUT_IDENTITIES_METADATA_ONLY.json'):sha(old/'FUTURE_INPUT_IDENTITIES_METADATA_ONLY.json')},
          'controls_for_gate':['zero','prefix_update_mean','same_information_Ridge'],
          'state_minimum_improvement_pct':.1,'task_minimum_improvement_pct':1.,'total_necessary_checks':26,
          'percentage_point_margin':1e-8,'independent_loss_rtol':1e-11,
          'numeric_identity_eps_factor':256,'fp64_error_factor':128,
          'runtime_seconds':180,'analysis_CPUs':[24,26],'GPU':False,'Q3_allowed':False,
          'new_market_fitting_allowed':False,'market_inference_allowed':False,
          'not_independent_market_confirmation':True}
    require(not (BASE/'DESIGN_SPEC.json').exists(),'Do not replace first spec')
    write(BASE/'DESIGN_SPEC.json',spec)

if __name__=='__main__': main()
