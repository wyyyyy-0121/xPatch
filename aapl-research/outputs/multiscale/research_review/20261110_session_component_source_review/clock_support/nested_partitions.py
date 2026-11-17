"""Pure clock/origin nested partitions and root-guarded artificial tests.

No Bars objects, model imports, price/label loaders, NPZ or network operations.
NumPy is imported lazily only inside the pure functions/artificial test entry.
The CLI accepts only --artificial-tests, never any data-file argument.
"""
from __future__ import annotations
from datetime import date, timedelta
import argparse
import hashlib
import json
import os

SOURCE_BINDINGS = {'original_inner_split': {'path': '/home/ayaka/wslcode/aapl-research/outputs/multiscale/controlled_followup/20261108_v2/code/reference_source/multiscale/splits.py', 'sha256': '456d8a0fe4ef667648bbd4f0235404bb824555643955da476b14e2729b7b2f7c', 'lines': [9, 20]}, 'original_P_T': {'path': '/home/ayaka/wslcode/aapl-research/outputs/multiscale/research_review/20261110_market_context_origin_coverage/coverage_core.py', 'sha256': '44e331dc4d94cf0da16b1d9225e83450066c9bb31f6446896457197316fff5b6', 'lines': [25, 40]}, 'proposed_nested_design': {'path': '/home/ayaka/wslcode/aapl-research/outputs/multiscale/research_review/20261110_session_component_source_review/TRAINING_DESIGN_REFINEMENT_ZH.md', 'sha256': 'c9c7501b3a3807f6d2d674065ed6c339c9b8d0860db73c600dbeb4d399d0c285'}}
ROLE_NAMES = ("Q_fit", "S_early_stop", "C_calibration", "gap_Q_to_S",
              "gap_B_to_C", "gap_P_to_T", "T_assessment")


class PartitionError(ValueError):
    def __init__(self, stage, reason):
        self.stage = stage
        self.reason = reason
        super().__init__(stage + ":" + reason)


def _require(condition, stage, reason):
    if not bool(condition):
        raise PartitionError(stage, reason)


def _readonly(array):
    result = array.copy()
    result.flags.writeable = False
    return result


def _array_hash(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _inner_split(train, ends, sessions, horizon, stage):
    """Literal original inner_split .15 grouping/purge, with explicit failure."""
    import numpy as np
    groups = np.unique(sessions[train])
    tail_count = max(1, int(np.ceil(len(groups) * 0.15)))
    _require(len(groups) > tail_count, stage, "insufficient_groups")
    stop_mask = np.isin(sessions[train], groups[-tail_count:])
    stop = train[stop_mask].copy()
    _require(len(stop) > 0, stage, "empty_stop")
    stop_start = ends[stop[0]]
    fit_mask = ends[train + horizon] < stop_start
    fit = train[fit_mask].copy()
    _require(len(fit) > 0, stage, "insufficient_fit")
    gap = train[~fit_mask & ~stop_mask].copy()
    return fit, stop, gap, {"group_count": len(groups), "stop_group_count": tail_count,
                           "stop_fraction": 0.15, "first_stop_decision_ns": int(stop_start)}


def build_nested_partitions(clock, fit, accepted_prefix_mask, accepted_tail_mask,
                            *, horizon, lookback):
    """Return immutable origin/position partitions plus an exhaustive ledger.

    clock must be exactly {end_ns: int64[n], session_date: Unicode[n]}.
    fit is the frozen int64 original bar-origin vector. The two accepted masks
    are bool[len(fit)] indexed by ORIGINAL FIT POSITION, not by bar index.
    No file is read. P/T is verified, not replaced. Only two original .15
    inner splits are added: P -> B/C_cal, then B -> Q/S.
    """
    import numpy as np
    _require(type(horizon) is int and horizon > 0, "input", "invalid_horizon")
    _require(type(lookback) is int and lookback > 0, "input", "invalid_lookback")
    _require(isinstance(clock, dict) and set(clock) == {"end_ns", "session_date"},
             "input", "clock_fields_must_be_end_ns_and_session_date_only")
    ends, sessions = clock["end_ns"], clock["session_date"]
    _require(isinstance(ends, np.ndarray) and ends.dtype == np.dtype("int64")
             and ends.ndim == 1 and len(ends) > horizon, "input", "invalid_end_clock")
    _require(isinstance(sessions, np.ndarray) and sessions.dtype.kind == "U"
             and sessions.ndim == 1 and sessions.shape == ends.shape,
             "input", "invalid_session_array")
    _require(np.all(ends[1:] > ends[:-1]), "input", "end_clock_not_strict")
    _require(np.all(sessions[1:] >= sessions[:-1]), "input", "session_order_not_monotonic")
    for value in np.unique(sessions):
        try:
            valid = date.fromisoformat(str(value)).isoformat() == str(value)
        except ValueError:
            valid = False
        _require(valid, "input", "session_date_not_canonical_ISO")
    _require(isinstance(fit, np.ndarray) and fit.dtype == np.dtype("int64")
             and fit.ndim == 1 and len(fit) >= 2, "input", "invalid_original_fit")
    _require(np.all(fit[1:] > fit[:-1]) and np.all(fit >= lookback-1)
             and np.all(fit <= len(ends)-1-horizon), "input", "fit_order_or_clock_bounds")
    for name, mask in (("prefix", accepted_prefix_mask), ("tail", accepted_tail_mask)):
        _require(isinstance(mask, np.ndarray) and mask.dtype == np.bool_
                 and mask.shape == fit.shape, "input", name + "_mask_not_original_fit_bool")
    cut = 4 * len(fit) // 5
    expected_tail = np.arange(len(fit)) >= cut
    T = fit[expected_tail]
    _require(len(T) > 0, "original_P_T", "empty_tail")
    expected_prefix = ((np.arange(len(fit)) < cut)
                       & (ends[fit + horizon] < ends[T[0]]))
    _require(np.array_equal(accepted_prefix_mask, expected_prefix),
             "original_P_T", "accepted_prefix_identity_mismatch")
    _require(np.array_equal(accepted_tail_mask, expected_tail),
             "original_P_T", "accepted_tail_identity_mismatch")
    P = fit[accepted_prefix_mask]
    _require(len(P) > 0, "original_P_T", "empty_prefix")
    gap_PT = fit[~accepted_prefix_mask & ~accepted_tail_mask]
    B, C_cal, gap_BC, pc_meta = _inner_split(P, ends, sessions, horizon, "P_to_B_Ccal")
    Q, S, gap_QS, qs_meta = _inner_split(B, ends, sessions, horizon, "B_to_Q_S")
    _require(len(Q) >= 2, "B_to_Q_S", "Q_requires_at_least_two_origins")
    for left, right, stage in ((Q, S, "Q_before_S"), (S, C_cal, "S_before_Ccal"),
                               (C_cal, T, "Ccal_before_T")):
        _require(int(ends[left[-1]+horizon]) < int(ends[right[0]]), stage, "strict_label_purge_failed")
    arrays = {"F": fit, "P": P, "B": B, "Q": Q, "S": S, "C_cal": C_cal, "T": T,
              "gap_P_to_T": gap_PT, "gap_B_to_C": gap_BC, "gap_Q_to_S": gap_QS}
    parents = {"F": None, "P": "F", "B": "P", "Q": "B", "S": "B", "C_cal": "P",
               "T": "F", "gap_P_to_T": "F", "gap_B_to_C": "P", "gap_Q_to_S": "B"}
    partitions = {}
    for name, origins in arrays.items():
        positions = np.searchsorted(fit, origins).astype(np.int64)
        parent = parents[name]
        parent_positions = positions if parent is None else np.searchsorted(arrays[parent], origins).astype(np.int64)
        _require(np.array_equal(fit[positions], origins), "ledger", "origin_position_mapping")
        if parent is not None:
            _require(np.array_equal(arrays[parent][parent_positions], origins), "ledger", "parent_position_mapping")
        partitions[name] = {"origins": _readonly(origins.astype(np.int64, copy=False)),
                            "fit_positions": _readonly(positions), "parent": parent,
                            "parent_positions": _readonly(parent_positions)}
    roles = np.full(len(fit), "", dtype="<U24")
    role_sources = {"Q_fit": Q, "S_early_stop": S, "C_calibration": C_cal,
                    "gap_Q_to_S": gap_QS, "gap_B_to_C": gap_BC,
                    "gap_P_to_T": gap_PT, "T_assessment": T}
    for role, origins in role_sources.items():
        positions = np.searchsorted(fit, origins)
        _require(np.all(roles[positions] == ""), "ledger", "overlapping_terminal_roles")
        roles[positions] = role
    _require(np.all(roles != ""), "ledger", "unassigned_original_origin")
    # Historical diagnostics never influence partition selection. Exclude the
    # global transition at the window's own first slot, exactly like g[0]=False.
    indices = np.arange(len(ends), dtype=np.int64)
    transitions = np.r_[False, sessions[1:] != sessions[:-1]]
    first_session = np.r_[True, sessions[1:] != sessions[:-1]]
    last_start = np.maximum.accumulate(np.where(first_session, indices, 0))
    window_start = fit - lookback + 1
    suffix_length = fit - np.maximum(window_start, last_start[fit]) + 1
    cumulative = np.cumsum(transitions, dtype=np.int64)
    boundary_count = cumulative[fit] - cumulative[window_start]
    support = {"origins": _readonly(fit),
               "fit_positions": _readonly(np.arange(len(fit), dtype=np.int64)),
               "suffix_length": _readonly(suffix_length.astype(np.int64)),
               "last_reset_local_position": _readonly((lookback-suffix_length).astype(np.int64)),
               "boundary_count": _readonly(boundary_count.astype(np.int64))}
    metadata = {"kind": "original_fit_matched_nested_clock_partitions_v1", "horizon": horizon,
                "lookback": lookback, "original_cut": cut, "original_P_T_preserved": True,
                "all_original_origins_accounted": True, "suffix_used_for_selection": False,
                "P_to_B_Ccal": pc_meta, "B_to_Q_S": qs_meta,
                "source_bindings": json.loads(json.dumps(SOURCE_BINDINGS)), "source_files_read_by_function": False,
                "array_identities": {name: {"count": len(p["origins"]),
                 "origins_int64_sha256": _array_hash(p["origins"]),
                 "fit_positions_int64_sha256": _array_hash(p["fit_positions"])} for name, p in partitions.items()},
                "original_prefix_mask_sha256": _array_hash(accepted_prefix_mask),
                "original_tail_mask_sha256": _array_hash(accepted_tail_mask),
                "clock_end_ns_sha256": _array_hash(ends), "session_date_sha256": _array_hash(sessions)}
    ledger = {"origins": _readonly(fit), "fit_positions": _readonly(np.arange(len(fit), dtype=np.int64)),
              "terminal_role": _readonly(roles)}
    return {"partitions": partitions, "ledger": ledger, "history_support": support, "metadata": metadata}


def artificial_tests():
    """Only in-memory artificial integer clocks and Unicode session labels."""
    import numpy as np
    passed = []
    def clock_from_counts(counts):
        stamps, sessions = [], []
        for i, count in enumerate(counts):
            day = date(2020, 1, 1) + timedelta(days=i)
            for slot in range(count):
                stamps.append((i * 86400 + slot * 60 + 1) * 1_000_000_000)
                sessions.append(day.isoformat())
        return {"end_ns": np.asarray(stamps, dtype=np.int64),
                "session_date": np.asarray(sessions, dtype="<U10")}
    def original_masks(clock, fit, H):
        cut = 4 * len(fit) // 5
        tail = np.arange(len(fit)) >= cut
        prefix = (np.arange(len(fit)) < cut) & (clock["end_ns"][fit+H] < clock["end_ns"][fit[cut]])
        return prefix, tail
    def run(clock, fit, H=1, L=1):
        pm, tm = original_masks(clock, fit, H)
        return build_nested_partitions(clock, fit, pm, tm, horizon=H, lookback=L)
    regular = clock_from_counts([3] * 30)
    fit = np.arange(89, dtype=np.int64)
    result = run(regular, fit)
    expected = {"F": range(89), "P": range(70), "B": range(59), "Q": range(50),
                "S": range(51, 59), "C_cal": range(60, 70), "T": range(71, 89),
                "gap_Q_to_S": [50], "gap_B_to_C": [59], "gap_P_to_T": [70]}
    for name, values in expected.items():
        _require(np.array_equal(result["partitions"][name]["origins"], np.array(list(values), dtype=np.int64)),
                 "artificial", "manual_regular_roles_" + name)
    _require(result["metadata"]["P_to_B_Ccal"]["group_count"] == 24
             and result["metadata"]["P_to_B_Ccal"]["stop_group_count"] == 4
             and result["metadata"]["B_to_Q_S"]["group_count"] == 20
             and result["metadata"]["B_to_Q_S"]["stop_group_count"] == 3,
             "artificial", "ceil_session_groups")
    passed.append("manual_30_sessions_ceil_and_all_exact_Q_S_Ccal_T_gap_origins")
    for gap, target in ((50, 51), (59, 60), (70, 71)):
        _require(regular["end_ns"][gap+1] == regular["end_ns"][target],
                 "artificial", "equality_fixture")
    passed.append("strict_label_end_equals_next_decision_is_purged_at_all_three_levels")
    uneven = clock_from_counts([9, 2, 5, 3, 1, 4, 2, 5] * 4)
    uf = np.arange(6, len(uneven["end_ns"])-1, dtype=np.int64)
    before = {k: v.copy() for k, v in uneven.items()}
    uneven_result = run(uneven, uf, L=7)
    _require(all(np.array_equal(uneven[k], before[k]) for k in uneven), "artificial", "input_mutation")
    support = uneven_result["history_support"]
    _require(np.any(support["suffix_length"] == 1) and np.any(support["suffix_length"] == 7),
             "artificial", "short_and_full_suffix_fixture")
    _require(np.array_equal(uneven_result["ledger"]["origins"], uf)
             and np.all(uneven_result["ledger"]["terminal_role"] != ""),
             "artificial", "suffix_must_not_filter_original_origins")
    for row, origin in enumerate(uf):
        history = uneven["session_date"][origin-6:origin+1]
        g = np.r_[False, history[1:] != history[:-1]]
        resets = np.flatnonzero(g)
        last = int(resets[-1]) if len(resets) else 0
        _require(support["suffix_length"][row] == 7-last and support["boundary_count"][row] == int(g.sum()),
                 "artificial", "suffix_bruteforce_history_semantics")
    for name, part in uneven_result["partitions"].items():
        _require(part["origins"].dtype == np.int64 and part["fit_positions"].dtype == np.int64
                 and part["parent_positions"].dtype == np.int64
                 and np.array_equal(uf[part["fit_positions"]], part["origins"]),
                 "artificial", "int64_original_source_positions_" + name)
        parent = part["parent"]
        if parent is not None:
            _require(np.array_equal(uneven_result["partitions"][parent]["origins"][part["parent_positions"]],
                                   part["origins"]), "artificial", "parent_positions_" + name)
        _require(not part["origins"].flags.writeable and not part["fit_positions"].flags.writeable,
                 "artificial", "immutable_partition_" + name)
    _require(set(uneven_result["ledger"]["terminal_role"].tolist()).issubset(set(ROLE_NAMES)),
             "artificial", "terminal_role_vocabulary")
    passed.append("unequal_half_day_like_sessions_exact_positions_immutable_inputs_and_suffix_no_filter")
    def rejects(name, func, expected_stage=None, expected_reason=None):
        try:
            func()
        except PartitionError as error:
            _require(expected_stage is None or error.stage == expected_stage, "artificial", "wrong_failure_stage_"+name)
            _require(expected_reason is None or error.reason == expected_reason, "artificial", "wrong_failure_reason_"+name)
            passed.append("reject_"+name)
        else:
            raise PartitionError("artificial", "invalid_case_accepted_"+name)
    one = clock_from_counts([40]); onefit = np.arange(39, dtype=np.int64)
    rejects("one_group_first_split", lambda: run(one, onefit), "P_to_B_Ccal", "insufficient_groups")
    two = clock_from_counts([20, 20]); twofit = np.arange(39, dtype=np.int64)
    rejects("one_remaining_group_second_split", lambda: run(two, twofit), "B_to_Q_S", "insufficient_groups")
    pm, tm = original_masks(regular, fit, 1)
    badpm = pm.copy(); badpm[0] = False
    badtm = tm.copy(); badtm[0] = True
    rejects("accepted_prefix_identity", lambda: build_nested_partitions(regular, fit, badpm, tm, horizon=1, lookback=1),
            "original_P_T", "accepted_prefix_identity_mismatch")
    rejects("accepted_tail_identity", lambda: build_nested_partitions(regular, fit, pm, badtm, horizon=1, lookback=1),
            "original_P_T", "accepted_tail_identity_mismatch")
    rejects("nonbool_mask", lambda: build_nested_partitions(regular, fit, pm.astype(np.int64), tm, horizon=1, lookback=1), "input")
    rejects("float_fit", lambda: build_nested_partitions(regular, fit.astype(np.float64), pm, tm, horizon=1, lookback=1), "input")
    rejects("extra_clock_field", lambda: build_nested_partitions({**regular, "unexpected": np.zeros(90)}, fit, pm, tm, horizon=1, lookback=1), "input")
    duplicate = {k: v.copy() for k, v in regular.items()}; duplicate["end_ns"][1] = duplicate["end_ns"][0]
    rejects("non_strict_clock", lambda: build_nested_partitions(duplicate, fit, pm, tm, horizon=1, lookback=1), "input")
    backwards = {k: v.copy() for k, v in regular.items()}; backwards["session_date"][8] = "2019-12-31"
    rejects("session_order", lambda: build_nested_partitions(backwards, fit, pm, tm, horizon=1, lookback=1), "input")
    rejects("history_out_of_bounds", lambda: build_nested_partitions(regular, fit, pm, tm, horizon=1, lookback=2), "input")
    return {"status": "artificial_tests_complete", "checks": passed,
            "manual_regular_fixture_partition_identities": result["metadata"]["array_identities"],
            "uneven_fixture_partition_identities": uneven_result["metadata"]["array_identities"],
            "market_files_read": False, "real_clock_run": False, "training_admitted": False}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artificial-tests", action="store_true", required=True,
                        help="Run only the in-memory clock fixtures; root owns the 30s protocol/receipt")
    parser.parse_args()
    _require(sorted(os.sched_getaffinity(0)) == [24], "artificial", "CPU24_required")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        _require(os.environ.get(key) == "1", "artificial", key+"_must_be_1")
    result = artificial_tests()
    result.update(CPU_affinity=sorted(os.sched_getaffinity(0)), numeric_threads=1,
                  source_bindings=SOURCE_BINDINGS, external_signal_recovery_tested=False)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
