"""Keep the existing completion checks across one explicitly signed cloud handoff."""
from datetime import datetime, timezone
import os
from pathlib import Path

from fusion.next.common import OUT, ROOT, digest, identity, read_json, write_json, utc_now
from scripts import finish_fusion_100h as finish


ORIGINAL_CHECK = finish.check_binding
ORIGINAL_OBSERVE = finish.OBSERVER


def extended_observer():
    source = ORIGINAL_OBSERVE
    source = source.replace("'dispatch':dispatch,", "'handoff':read('cloud/speed_handoff_state.json'), 'dispatch':dispatch,")
    return source


def check(snapshot, binding):
    request = read_json(OUT/'cloud/speed_handoff_request.json')
    if request['identity'] != identity({k:v for k,v in request.items() if k != 'identity'}):
        raise ValueError('Local handoff request changed')
    if digest(Path(__file__)) != request['completion_wrapper_sha256']:
        raise ValueError('Handoff completion wrapper changed')
    if snapshot['limit'] != request['limit'] or binding['limit'] != request['limit']:
        raise ValueError('Handoff cannot alter the time limit')
    if binding['source_identity'] != request['source_identity']:
        raise ValueError('Handoff cannot alter the engine')
    if binding['scope_identity'] != request['old_dispatch']['scope_identity']:
        raise ValueError('Handoff cannot alter the research scope')
    phase = snapshot.get('handoff')
    if phase and phase['request_identity'] != request['identity']:
        raise ValueError('Remote handoff identity differs')
    if phase and phase['phase'] == 'failed':
        raise RuntimeError('Cloud handoff failed: '+phase.get('error','unknown'))
    dispatch = snapshot['dispatch']
    if dispatch == binding['dispatch']:
        # After rebind all original rules apply again; never excuse future exits.
        if dispatch.get('handoff_request_identity') == request['identity']:
            return ORIGINAL_CHECK(snapshot, binding)
        if not phase or phase['phase'] == 'waiting_for_stage':
            return ORIGINAL_CHECK(snapshot, binding)
    else:
        if (dispatch.get('handoff_request_identity') != request['identity'] or
                dispatch.get('previous_dispatch') != request['old_dispatch'] or
                dispatch.get('speed_policy_identity') != request['speed_policy_identity']):
            raise ValueError('Unapproved new cloud runner')
        if phase and phase['phase'] == 'complete':
            if phase.get('new_dispatch') != dispatch or not phase.get('old_runner_confirmed_exited'):
                raise ValueError('Completed handoff lacks exact exit/dispatch evidence')
            if not snapshot['runner_alive'] or not snapshot['limiter_alive']:
                raise RuntimeError('Handoff runner or hard limiter is not alive')
            old = dict(binding)
            saved = finish.HERE/'before_speed_handoff_binding.json'
            if saved.exists() and read_json(saved) != old:
                raise ValueError('Prior handoff binding differs')
            write_json(saved, old)
            binding.update(dispatch=dispatch, speed_policy_identity=request['speed_policy_identity'],
                           handoff_request_identity=request['identity'],
                           handoff_rebound_at_utc=utc_now())
            write_json(finish.HERE/'binding.json',binding)
            return ORIGINAL_CHECK(snapshot, binding)
    if not phase or phase['phase'] not in ('stopping_old_runner','starting_fast_runner'):
        raise ValueError('Unexpected cloud identity transition')
    elapsed = (datetime.now(timezone.utc)-datetime.fromisoformat(phase['at_utc'])).total_seconds()
    if not 0 <= elapsed <= 180:
        raise TimeoutError('Planned cloud handoff exceeded its bounded transition window')
    if not snapshot['limiter_alive']:
        raise RuntimeError('Hard limiter missing during handoff')
    if snapshot.get('numerical_policy',{}).get('identity') != request['numerical_policy_identity']:
        raise ValueError('Handoff numerical policy changed')
    if snapshot['state']['source_identity'] != request['source_identity'] or snapshot['state']['scope_identity'] != binding['scope_identity']:
        raise ValueError('Handoff state identity changed')
    if snapshot['state']['stage'] == 'stopped_for_inspection' and snapshot['state'].get('error_type') != 'KeyboardInterrupt':
        raise RuntimeError('Unplanned cloud failure during handoff')
    return False


def run():
    from scripts.portable_inference_runtime import install
    install()
    bootstrap = ROOT/'tmp/portable_inference_bootstrap'
    os.environ['PYTHONPATH'] = str(bootstrap)+os.pathsep+str(ROOT)
    finish.OBSERVER = extended_observer()
    finish.check_binding = check
    finish.run()


if __name__ == '__main__':
    run()
