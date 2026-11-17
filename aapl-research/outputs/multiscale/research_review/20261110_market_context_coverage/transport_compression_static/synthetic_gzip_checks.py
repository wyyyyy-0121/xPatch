"""Offline synthetic gzip checks only. Contains no URL/authentication/network calls."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import zlib

CAP = 4 * 1024 * 1024
CHUNK = 65536

class Rejected(ValueError):
    pass


def decode_bounded(encoded: bytes, coding: str | None):
    # Prototype for testing a design, not an admitted acquisition adapter.
    if len(encoded) > CAP:
        raise Rejected('encoded_cap')
    coding = 'identity' if coding is None else coding.strip().lower()
    if coding == 'identity':
        return encoded, len(encoded)
    if coding != 'gzip':
        raise Rejected('unsupported_or_stacked_encoding')
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    result = bytearray()
    max_piece = 0
    for offset in range(0, len(encoded), CHUNK):
        pending = encoded[offset:offset + CHUNK]
        while pending:
            before = len(pending)
            piece = decoder.decompress(pending, min(CHUNK, CAP - len(result) + 1))
            max_piece = max(max_piece, len(piece))
            result.extend(piece)
            if len(result) > CAP:
                raise Rejected('decoded_cap_guard')
            pending = decoder.unconsumed_tail
            if decoder.eof:
                if decoder.unused_data or pending or offset + CHUNK < len(encoded):
                    raise Rejected('trailing_or_multimember')
                break
            if not piece and len(pending) == before:
                raise Rejected('decoder_no_progress')
    if not decoder.eof:
        raise Rejected('truncated_gzip')
    if decoder.unused_data or decoder.unconsumed_tail:
        raise Rejected('trailing_or_unconsumed')
    return bytes(result), max_piece


def fixture(count=10000):
    start = datetime(2021, 1, 4, tzinfo=timezone.utc)
    bars = [{'t': (start + timedelta(minutes=i)).isoformat().replace('+00:00', 'Z'),
             'o': 1.0, 'h': 1.0, 'l': 1.0, 'c': 1.0, 'v': 0, 'n': 0, 'vw': 1.0}
            for i in range(count)]
    return json.dumps({'bars': {'SPY': bars}, 'next_page_token': None},
                      separators=(',', ':')).encode('ascii')


def main():
    if os.sched_getaffinity(0) != {26}:
        raise RuntimeError('CPU26_required')
    checks=[]
    def good(name, test):
        if not test():
            raise AssertionError(name)
        checks.append({'name': name, 'passed': True})
    def bad(name, data, coding='gzip'):
        try:
            decode_bounded(data, coding)
        except (Rejected, zlib.error):
            checks.append({'name': name, 'passed': True})
        else:
            raise AssertionError(name)
    plain=fixture()
    packed=gzip.compress(plain, compresslevel=6, mtime=0)
    good('valid_single_member_exact_decoded_identity', lambda: decode_bounded(packed, 'gzip')[0] == plain)
    good('missing_encoding_identity', lambda: decode_bounded(plain, None)[0] == plain)
    good('explicit_identity', lambda: decode_bounded(plain, 'identity')[0] == plain)
    good('case_insensitive_gzip_token', lambda: decode_bounded(packed, ' GZip ')[0] == plain)
    for encoding in ('br', 'deflate', 'gzip,identity', 'gzip,gzip', '', 'x-gzip'):
        bad('reject_encoding_' + repr(encoding), packed, encoding)
    bad('claimed_gzip_receives_identity', plain)
    wrong=bytearray(packed); wrong[-8] ^= 1
    bad('CRC32_mismatch', bytes(wrong))
    wrong=bytearray(packed); wrong[-4] ^= 1
    bad('ISIZE_mismatch', bytes(wrong))
    bad('truncated_footer', packed[:-1])
    bad('truncated_body', packed[:len(packed)//2])
    bad('concatenated_gzip_members', packed+packed)
    bad('trailing_nonmember_bytes', packed+b'garbage')
    bad('trailing_zero_padding', packed+b'\x00')
    good('exact_decoded_cap_accepted', lambda: len(decode_bounded(gzip.compress(b'x'*CAP, mtime=0), 'gzip')[0]) == CAP)
    bad('decoded_cap_plus_one_rejected', gzip.compress(b'x'*(CAP+1), mtime=0))
    bad('zip_bomb_16MiB_rejected_at_4MiB_guard', gzip.compress(b'x'*(4*CAP), mtime=0))
    bad('encoded_cap_plus_one_rejected', b'x'*(CAP+1), 'identity')
    good('per_piece_decoding_limit', lambda: decode_bounded(packed, 'gzip')[1] <= CHUNK)
    good('conservative_max_charge_within_original_reservation', lambda: max(len(packed), len(plain)) <= CAP + 1)
    good('identity_ledger_unchanged', lambda: max(len(plain), len(plain)) == len(plain))
    good('budget_max_charge_bounds_both_totals', lambda: sum(max(w,d) for w,d in [(100,1000),(5000,4000)]) >= max(100+5000,1000+4000))
    for _ in range(5):
        decode_bounded(packed, 'gzip')
    wall=[]; cpu=[]
    for _ in range(100):
        w=time.perf_counter(); c=time.process_time()
        decoded,_=decode_bounded(packed, 'gzip')
        cpu.append(time.process_time()-c); wall.append(time.perf_counter()-w)
        if decoded != plain:
            raise AssertionError('benchmark_identity')
    receipt={'status':'offline_synthetic_design_checks_only_not_live_admission',
             'CPU_affinity':sorted(os.sched_getaffinity(0)),
             'HTTP_calls':0, 'credentials_accessed':False,
             'actual_market_rows_arrays_checkpoints_read':False,
             'current_acquisition_raw_or_journal_read':False,
             'fixture':'10000 artificial minute timestamps, all prices=1.0, volume/trades=0; no market values',
             'fixture_sha256':hashlib.sha256(plain).hexdigest(),
             'fixture_plain_bytes':len(plain),'fixture_gzip_bytes':len(packed),
             'synthetic_ratio_encoded_over_decoded':len(packed)/len(plain),
             'gzip_compresslevel':6, 'gzip_mtime':0,
             'decoded_page_cap_bytes':CAP,
             'iterations':len(cpu),
             'median_decode_CPU_seconds':statistics.median(cpu),
             'maximum_decode_CPU_seconds':max(cpu),
             'median_decode_wall_seconds':statistics.median(wall),
             'maximum_decode_wall_seconds':max(wall),
             'synthetic_results_do_not_predict_supplier_compression_or_speed':True,
             'checks_passed':len(checks), 'checks':checks,
             'fixture_runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dest=Path(__file__).with_name('SYNTHETIC_GZIP_CHECKS.json')
    dest.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','checks_passed','fixture_plain_bytes','fixture_gzip_bytes','synthetic_ratio_encoded_over_decoded','median_decode_CPU_seconds','maximum_decode_CPU_seconds')}))

if __name__ == '__main__':
    main()
