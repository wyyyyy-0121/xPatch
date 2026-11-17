"""Only generated NPZs/JSONs in a temporary directory, never real market data."""
import calendar
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import types
from unittest.mock import patch
import warnings
import zipfile

import numpy as np

import coverage_core as core
import coverage_worker as worker


def main():
    started = time.time(); checks = []
    def check(name, value):
        if not bool(value):
            raise AssertionError(name)
        checks.append(name)
    def reject(name, fn):
        try:
            fn()
        except (ValueError, RuntimeError):
            checks.append(name)
        else:
            raise AssertionError(name)
    minute = 60_000_000_000
    with tempfile.TemporaryDirectory(prefix='synthetic-clock-coverage-') as tmp:
        tmp = Path(tmp)
        starts = np.r_[np.arange(390)*minute, (1440+np.arange(210))*minute].astype(np.int64)
        dates = np.array(['2021-01-04']*390+['2021-01-05']*210)
        bars = {'start_ns': starts, 'end_ns': starts+minute,
                'observed': np.ones(600, dtype=bool), 'session_date': dates}
        path = tmp/'bars.npz'
        # The poison fields cannot be loaded with allow_pickle=False.
        np.savez_compressed(path, **bars, close=np.array([{'forbidden': True}], dtype=object),
                            labels=np.array([{'forbidden': True}], dtype=object))
        opened = []; original_open = zipfile.ZipFile.open
        def audited_open(archive, name, *args, **kwargs):
            opened.append(name)
            return original_open(archive, name, *args, **kwargs)
        audit = []
        with patch.object(zipfile.ZipFile, 'open', audited_open):
            read = worker.read_selected_members(path, worker.BAR_MEMBERS, audit)
        check('only_four_clock_members_opened', opened == [x+'.npy' for x in worker.BAR_MEMBERS])
        check('poison_close_labels_not_decoded', set(read) == set(worker.BAR_MEMBERS))
        check('read_audit_is_actual_member_list', [a['member'] for a in audit] == opened)
        reject('close_selection_rejected', lambda: worker.read_selected_members(path, ('close',), []))
        split = tmp/'split.npz'
        np.savez_compressed(split, fit=np.arange(10, dtype=np.int64),
            stop=np.array([{'forbidden': True}], dtype=object),
            validation=np.array([{'forbidden': True}], dtype=object))
        audit = []; opened.clear()
        with patch.object(zipfile.ZipFile, 'open', audited_open):
            selected = worker.read_selected_members(split, ('fit',), audit)
        check('fit_only_no_stop_outer_open', opened == ['fit.npy'] and set(selected) == {'fit'})
        duplicate = tmp/'duplicate.npz'
        member = io.BytesIO(); np.save(member, np.arange(2, dtype=np.int64), allow_pickle=False)
        with zipfile.ZipFile(duplicate, 'w') as archive, warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            archive.writestr('fit.npy', member.getvalue()); archive.writestr('fit.npy', member.getvalue())
        reject('duplicate_selected_member_rejected', lambda: worker.read_selected_members(duplicate, ('fit',), []))
        worker.check_bar_clock(bars, 600, minute)
        check('valid_development_clock', True)
        wrong = dict(bars, end_ns=np.full(600, worker.CUTOFF_NS+minute, dtype=np.int64))
        reject('Q3_clock_rejected', lambda: worker.check_bar_clock(wrong, 600, minute))
        a, b = core.original_hour_groups(starts, starts+minute, dates)
        hour = {'start_ns': starts[a], 'end_ns': (starts+minute)[b-1],
                'session_date': dates[a], 'observed': np.ones(len(a), dtype=bool)}
        actual_a, actual_b = worker.check_original_hour_clock(bars, hour)
        check('full_hour_grid_exact_match', np.array_equal(a, actual_a) and np.array_equal(b, actual_b))
        wrong = copy.deepcopy(hour); wrong['start_ns'][6] = starts[360]
        reject('in_bounds_cross_session_hour_rejected', lambda: worker.check_original_hour_clock(bars, wrong))
        wrong = copy.deepcopy(hour); wrong['session_date'][6] = '2021-01-04'
        reject('wrong_hour_session_rejected', lambda: worker.check_original_hour_clock(bars, wrong))
        src = np.array([-minute, 0, minute//2, minute, starts[-1]+minute], dtype=np.int64)
        observed, outside = worker.map_exact_source(starts, src)
        check('exact_timestamp_only_no_nearest', np.flatnonzero(observed).tolist() == [0, 1])
        check('out_of_session_not_grid_compressed', len(observed) == 600 and outside == 3)
        reject('unordered_source_rejected', lambda: worker.map_exact_source(starts, src[::-1]))
        reject('duplicate_source_rejected', lambda: worker.map_exact_source(starts, np.array([0, 0], dtype=np.int64)))

        # One artificial bar in each original range exercises all index/token/hash bounds.
        body = worker.SOURCE.read_bytes()
        source = types.ModuleType('fixture_source'); source.__file__ = str(worker.SOURCE)
        exec(compile(body, str(worker.SOURCE), 'exec'), source.__dict__)
        index = {'kind': 'SPY_qualified_canonical_index_v1', 'all_67_ranges_complete': True,
            'ranges': source.RANGES, 'Q3_allowed': False,
            'source_only_not_RTH_or_context_admission': True,
            'protocol_sha256': 'fixture-protocol', 'inheritance_manifest_sha256': 'fixture-manifest', 'pages': []}
        expected = []
        for r in range(67):
            low, _ = source.bounds(r); stamp = low.isoformat()
            payload = {'bars': {'SPY': [{'t': stamp, 'o': 1, 'h': 1, 'l': 1,
                'c': 1, 'v': 1, 'n': 1, 'vw': 1}]}, 'next_page_token': None}
            body = json.dumps(payload).encode(); path = tmp/(str(r)+'.json'); path.write_bytes(body)
            index['pages'].append({'range': r, 'page': 0, 'canonical_ref':
                {'path': str(path), 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()},
                'request_page_token': None, 'next_page_token': None, 'rows': 1,
                'first_UTC': stamp, 'last_UTC': stamp})
            expected.append(calendar.timegm(low.utctimetuple())*1_000_000_000)
        protocol = {'source_protocol_sha256': 'fixture-protocol', 'inheritance_manifest_sha256': 'fixture-manifest'}
        actual = worker.qualified_source_starts(index, protocol, lambda: None)
        check('all_67_artificial_pages_exact_UTC_ns', actual.tolist() == expected)
        wrong = copy.deepcopy(index); wrong['pages'][4]['request_page_token'] = 'future'
        reject('index_request_token_change_rejected', lambda: worker.qualified_source_starts(wrong, protocol, lambda: None))
        wrong = copy.deepcopy(index); wrong['pages'][0]['canonical_ref']['sha256'] = '0'*64
        reject('source_hash_change_rejected', lambda: worker.qualified_source_starts(wrong, protocol, lambda: None))
        wrong = copy.deepcopy(index); wrong['pages'] = wrong['pages'][:-1]
        reject('incomplete_source_index_rejected', lambda: worker.qualified_source_starts(wrong, protocol, lambda: None))
        wrong = copy.deepcopy(index); wrong['pages'][-1]['range'] = 67
        reject('outside_67_range_rejected', lambda: worker.qualified_source_starts(wrong, protocol, lambda: None))
        wrong = copy.deepcopy(index); wrong['all_67_ranges_complete'] = False
        reject('partial_cache_not_qualified', lambda: worker.qualified_source_starts(wrong, protocol, lambda: None))
    report = {'passed': True, 'checks_count': len(checks), 'checks': checks,
        'worker_sha256': worker.file_sha(Path(worker.__file__)),
        'test_sha256': worker.file_sha(Path(__file__)), 'seconds': time.time()-started,
        'actual_market_files_read': False, 'actual_prices_labels_Q3_read': False,
        'artificial_SPY_JSON_fields_decoded': True, 'HTTP_calls': 0, 'fits': 0, 'GPU': False}
    (Path(__file__).parent/'SYNTHETIC_WORKER_CHECKS.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed': True, 'checks_count': len(checks), 'seconds': report['seconds']}))


if __name__ == '__main__':
    main()
