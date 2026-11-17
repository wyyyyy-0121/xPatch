"""Explicit, bounded OHLC source qualification; never constructs labels."""
from pathlib import Path
import hashlib
import json
import os
import signal
import tempfile
import time

BASE = Path(__file__).resolve().parents[1]
STOP = False


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic(path, value):
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno()); temporary = handle.name
    os.replace(temporary, path)


def stop(*_):
    global STOP
    STOP = True


def main():
    protocol_path = BASE/'SOURCE_PROTOCOL.json'
    protocol = json.loads(protocol_path.read_text()); identity = sha(protocol_path)
    started = time.monotonic()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, stop)
    def guard():
        if STOP or time.time() >= protocol['deadline_epoch']:
            raise InterruptedError('Same frozen source deadline or stop request')
    try:
        guard()
        if os.sched_getaffinity(0) != {24}:
            raise ValueError('Use only CPU24 for source qualification')
        for path, expected in protocol['dependencies'].items():
            guard()
            if sha(path) != expected:
                raise ValueError('Source qualification dependency changed: '+path)
        if protocol['real_source_qualification_admitted'] is not True:
            raise ValueError('Source-only execution was not admitted')
        from ohlc_source import load_ohlc
        values = load_ohlc(BASE/'source_cache', guard)
        import numpy as np
        # Descriptive source-only geometry, not predictive evidence or masking.
        summaries = {}
        for frequency in ('minute','hour'):
            guard(); bars = values[frequency]; clv = bars['clv']
            summaries[frequency] = dict(count=len(clv), finite=bool(np.isfinite(clv).all()),
                min=float(clv.min()), max=float(clv.max()),
                flat_range_count=int((bars['high'] == bars['low']).sum()),
                earliest_bar_start_ns=int(bars['start_ns'][0]),
                last_bar_end_ns=int(bars['end_ns'][-1]))
        guard()
        atomic(BASE/'SOURCE_QUALIFICATION.json', dict(status='qualified_historical_final_bar_OHLC',
            protocol_sha256=identity, receipt_sha256=sha(BASE/'source_cache/OHLC_SOURCE_RECEIPT.json'),
            summaries=summaries, elapsed_seconds=time.monotonic()-started,
            actual_receipt_latency_or_revision_PIT_established=False,
            targets_or_models_loaded=False, predictive_effect_measured=False,
            Q3_read=False, GPU=False, new_market_candidate_admitted=False))
    except BaseException as error:
        atomic(BASE/'SOURCE_RUN_FAILURE.json', dict(protocol_sha256=identity,
            error_type=type(error).__name__, reason=str(error), elapsed_seconds=time.monotonic()-started,
            deadline_epoch=protocol['deadline_epoch'], qualified=False,
            partial_atomic_caches_preserved=True))
        raise


if __name__ == '__main__':
    import sys
    if sys.argv[1:] != ['--run']:
        raise SystemExit('Only explicit --run with frozen source-only protocol is supported')
    main()
