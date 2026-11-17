"""Reviewed-source reader for one frozen pre-Q3 development execution.

Physical decoding includes entire approved pre-Q3 close columns. Role access is
enforced by CommonAdapter, not by hiding all other timestamps from memory.
"""
from __future__ import annotations
import hashlib
import io
import json
import math
from pathlib import Path
import time
import zipfile
import numpy as np
from common_adapter import CommonAdapter, GlobalPhaseGate, StateKey, array_id, frozen

BAR_MEMBERS = ('start_ns', 'end_ns', 'observed', 'session_date', 'close')
ROLE_MEMBERS = ('Q_origins', 'S_origins', 'C_cal_origins', 'T_origins')
EXPECTED_STATES = (('minute_h60_future_log_return', 1), ('minute_h60_future_log_return', 2),
                   ('hourly_h8_future_log_return', 1), ('hourly_h8_future_log_return', 2))
SCIENCE_SHA256 = 'e9e562d5b6f9d795104389f9d91950a737765566d041524dd8cd27360a1ba12b'
CUTOFF = 1782878400000000000


class InputError(ValueError):
    pass


def require(value, message):
    if not bool(value):
        raise InputError(message)


class MarketInputs:
    def __init__(self, protocol_path):
        self.protocol_path = Path(protocol_path).resolve()
        self.protocol_bytes = self.protocol_path.read_bytes()
        self.protocol = json.loads(self.protocol_bytes)
        p = self.protocol
        require(p['kind'] == 'matched_two_state_market_execution_v1', 'Execution protocol kind')
        require(p['market_price_io_allowed'] is True and p['Q3_allowed'] is False
                and p['GPU'] is False and p['paid_resources'] is False, 'Execution scope')
        require(all(type(p[k]) in (int, float) and math.isfinite(p[k])
                    for k in ('started_epoch', 'deadline_epoch'))
                and p['deadline_epoch']-p['started_epoch'] == 14400, 'Original four-hour window')
        self.guard()
        self.audit = []
        refs_body = self.read_verified(p['input_reference_path'], p['input_reference_sha256'])
        refs = json.loads(refs_body)
        require(refs['kind'] == 'two_state_market_source_and_screen_state_references_proposal_v1'
                and refs['Q3_allowed'] is False and refs['GPU'] is False, 'Reference identity/scope')
        require(tuple(refs['future_price_reader_decode_allowlist']) == BAR_MEMBERS,
                'Exact source member allowlist')
        science = self.read_verified(p['science_rules_path'], SCIENCE_SHA256)
        require(p['science_rules_sha256'] == SCIENCE_SHA256
                and json.loads(science)['candidate'] == 'R_shrink', 'Frozen science definition')
        require(p['reader_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'Reader source identity')
        rows = refs['screen_states']
        require(tuple((r['task'], r['fold']) for r in rows) == EXPECTED_STATES
                and len({r['stateUID'] for r in rows}) == 4, 'Exact four ordered states')
        self.refs = refs
        self.states = tuple(StateKey(r['task'], r['fold'], r['stateUID']) for r in rows)
        self.gate = GlobalPhaseGate(self.states)
        self._bars = {}
        self._adapters = {}
        # Identity-only inputs. These operations remain inside the same window.
        for name in ('original_data', 'original_admission', 'clock_acceptance'):
            ref = refs[name]
            verified = self.read_verified(ref['path'], ref['sha256'])
            if name == 'clock_acceptance':
                require(json.loads(verified)['status'] ==
                        'accepted_all10_nested_roles_and_crossstate_decision_clock_manifest',
                        'Accepted nested role qualification')

    def guard(self):
        now = time.time()
        require(self.protocol['started_epoch'] <= now < self.protocol['deadline_epoch'],
                'Original absolute execution window expired or not started')

    def read_verified(self, path, expected_sha256):
        self.guard()
        value = Path(path).read_bytes()
        require(hashlib.sha256(value).hexdigest() == expected_sha256, 'Source checksum: '+str(path))
        self.guard()
        return value

    def selected(self, path, expected_sha256, names):
        require(names in (BAR_MEMBERS, ROLE_MEMBERS), 'Member decode allowlist')
        body = self.read_verified(path, expected_sha256)
        result = {}
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            listing = archive.namelist()
            for name in names:
                self.guard()
                member = name+'.npy'
                require(listing.count(member) == 1, 'Missing/duplicate NPZ member: '+member)
                with archive.open(member) as stream:
                    result[name] = np.lib.format.read_array(stream, allow_pickle=False)
                self.audit.append({'path': str(Path(path).resolve()), 'decoded_member': member,
                                   'decoded_epoch': time.time()})
        return result

    def bars(self, frequency):
        require(frequency in ('minute', 'hour'), 'Only original two frequencies')
        if frequency not in self._bars:
            ref = self.refs['original_bars'][frequency]
            values = self.selected(ref['path'], ref['sha256'], BAR_MEMBERS)
            n = ref['rows']
            require(all(a.ndim == 1 and a.shape == (n,) for a in values.values()), 'Bar shapes')
            require(values['start_ns'].dtype == np.int64 and values['end_ns'].dtype == np.int64
                    and values['observed'].dtype == np.bool_
                    and values['session_date'].dtype.kind == 'U'
                    and values['close'].dtype == np.float64, 'Bar dtypes')
            require(np.all(np.diff(values['start_ns']) > 0)
                    and np.all(np.diff(values['end_ns']) > 0)
                    and np.all(values['end_ns'] > values['start_ns'])
                    and np.all(values['end_ns'] <= CUTOFF)
                    and np.all(values['session_date'] <= '2026-06-30'), 'Pre-Q3 ordered clock')
            require(values['observed'].all() and np.isfinite(values['close']).all()
                    and np.all(values['close'] > 0), 'Source incomplete: no filling/filtering permitted')
            self._bars[frequency] = {k: frozen(v) for k, v in values.items()}
        self.guard()
        return dict(self._bars[frequency])

    def adapter(self, task, fold):
        require((task, fold) in EXPECTED_STATES, 'Unknown task/fold')
        key = (task, fold)
        if key not in self._adapters:
            i = EXPECTED_STATES.index(key); row = self.refs['screen_states'][i]
            roles = self.selected(row['nested_file'], row['nested_file_sha256'], ROLE_MEMBERS)
            checked = {}
            for name in ('Q', 'S', 'C_cal', 'T'):
                values = roles[name+'_origins']; meta = row['partitions'][name]
                require(values.dtype == np.int64 and values.ndim == 1
                        and len(values) == meta['count']
                        and array_id(values) == meta['origin_sha256'], 'Frozen role identity: '+name)
                checked[name] = values
            b = self.bars(row['frequency'])
            self._adapters[key] = CommonAdapter(self.states[i], self.gate, close=b['close'],
                end_ns=b['end_ns'], session_date=b['session_date'], roles=checked,
                split_actions=tuple(self.refs['split_actions_projection']))
        self.guard()
        return self._adapters[key]
