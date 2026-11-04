"""G1: independently verify raw receipts, labels, common origins and date partitions."""
import time
import numpy as np

from .common import ROOT, digest, runtime, utc_now, write_json
from .data import Market, development_splits


def main():
    runtime()
    started = time.perf_counter()
    market = Market.load(full_receipt=True)
    origins = market.origins
    np.testing.assert_allclose(market.y(origins), market.y_steps(origins).sum(axis=1), rtol=1e-10, atol=2e-15)
    assert np.isfinite(market.y_steps(origins)).all()
    assert np.all(market.dates[origins - 240] == market.dates[origins + 12])
    # Explicit last-observation/target endpoints and all eligible origin identities.
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "common_origins.npz", row=origins,
                        bar_start_ns=market.bars.index[origins].as_unit("ns").asi8,
                        label_end_ns=market.bars.index[origins + 12].as_unit("ns").asi8)
    splits = development_splits(market)
    for i in splits['inner']:
        assert max(i['fit_dates']) < min(i['purge_dates']) < min(i['validation_dates'])
        assert set(i['fit_dates']).isdisjoint(i['validation_dates'] + i['purge_dates'])
        assert len(i['purge_dates']) == 1
    d = splits['confirmation']
    assert max(d['fit_dates']) < min(d['purge_dates']) < min(d['evaluation_dates'])
    assert set(splits['final_fit_dates']).isdisjoint(splits['calibration_dates'])
    assert max(splits['final_fit_dates']) <= '2026-05-31'
    assert len(splits['calibration_dates']) == 20
    # Information-only perturbation: future values cannot enter historical window/scaler.
    fit_dates = sorted(set(market.dates))[:20]
    scale = market.scale(fit_dates)
    origin = market.select(fit_dates)[100]
    x = market.x([origin], 240, scale)
    saved = market.features[origin + 1:].copy()
    market.features[origin + 1:] = 12345.0
    np.testing.assert_array_equal(x, market.x([origin], 240, scale))
    market.features[origin + 1:] = saved
    local = market.bars.index[origins].tz_convert("America/New_York")
    primary = np.concatenate([market.select(i['validation_dates']) for i in splits['inner']])
    record = {"passed": True, "completed_at_utc": utc_now(), "wall_seconds": time.perf_counter() - started,
              **market.audit, "checks": {"direct_ratio_equals_sum_of_12_returns": True,
              "all_context_and_target_in_same_contiguous_session": True,
              "future_features_cannot_change_past_window": True,
              "development_dates_and_purges_disjoint": True, "shared_origins_all_lookbacks": True},
              "first_origin_bar_start_et": min(local.strftime("%H:%M")),
              "last_origin_bar_start_et": max(local.strftime("%H:%M")),
              "primary_origins": len(primary), "primary_effective_sessions": len(set(market.dates[primary])),
              "common_origins_sha256": digest(out / "common_origins.npz"),
              "plan_sha256": digest(ROOT / 'IMPLEMENTATION_PLAN.md'),
              "note": "No research model scores calculated. Development data and fixed date blocks admitted."}
    write_json(out / "development_splits.json", splits)
    record["development_splits_sha256"] = digest(out / "development_splits.json")
    write_json(out / "data_admission.json", record)
    print({k: record[k] for k in ("passed", "wall_seconds", "regular_rows", "sessions", "eligible_origins", "primary_origins")}, flush=True)


if __name__ == "__main__":
    main()
