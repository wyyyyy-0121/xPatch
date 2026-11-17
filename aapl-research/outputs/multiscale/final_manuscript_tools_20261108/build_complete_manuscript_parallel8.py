"""Build a complete manuscript only from the verified 100-fold evidence release.

This is a local editorial utility. It never trains a model, changes an original
receipt, or turns exposed development observations into independent evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics

MODELS = ("last_close", "drift", "seasonal", "ridge", "mlp", "gru", "lstm", "xpatch")
NAMES = ("Persistence / zero", "Drift", "Seasonal", "Ridge", "MLP", "GRU", "LSTM", "xPatch")
NEURAL = MODELS[4:]
SCALES = ("annual", "quarterly", "monthly", "hourly", "minute")
TASKS = (("annual", 12), ("quarterly", 3), ("monthly", 21),
         ("hourly", 1), ("hourly", 4), ("hourly", 8),
         ("minute", 5), ("minute", 15), ("minute", 30), ("minute", 60))


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def once(text, before, after):
    require(text.count(before) == 1, "Missing or ambiguous editorial fragment: " + before[:100])
    return text.replace(before, after, 1)


def section(text, start, end, body):
    require(text.count(start) == text.count(end) == 1, "Ambiguous section boundary")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return head + start + "\n\n" + body.strip() + "\n\n" + end + tail


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |"] +
                     ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def num(value):
    return "—" if value is None else f"{value:.6g}"


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def label(task):
    return task["profile"].title() + f" H{task['horizon']}" + (
        " (short history)" if task["profile"] == "annual" else "")


def validate(evidence):
    receipt = load(evidence / "ALL100_EVIDENCE_RECEIPT.json")
    require(receipt["status"] == "all100_evidence_ready_for_final_manuscript" and
            receipt["completed_folds"] == 100 and receipt["complete_tasks"] == 20 and
            receipt["Q3_values_read"] is False, "Full evidence release is not ready")
    for relative, expected in receipt["artifacts"].items():
        require(sha(evidence / relative) == expected, "Evidence changed: " + relative)
    results = load(evidence / "results/RESULTS.json")
    protocol = load(evidence / "results/COMPILATION_PROTOCOL.json")
    require(results["status"] == "complete_historical_development_results" and
            results["completed_folds"] == 100 and results["complete_tasks"] == 20 and
            len(results["tasks"]) == 20 and results["independent_test"] is False and
            results["p_values_reported"] is False and results["Q3_values_read"] is False,
            "Results are partial or have a different evidential scope")
    require(protocol["allow_partial"] is False and results["protocol_sha256"] ==
            sha(evidence / "results/COMPILATION_PROTOCOL.json"), "Compiler protocol differs")
    for name, expected in results["csv_sha256"].items():
        require(sha(evidence / "results" / name) == expected, "Result CSV changed: " + name)
    tasks = sorted(results["tasks"], key=lambda t: (SCALES.index(t["profile"]), t["horizon"], t["target"]))
    require({(t["profile"], t["horizon"]) for t in tasks} == set(TASKS), "Task set differs")
    for t in tasks:
        require(t["completed_folds"] == [1, 2, 3, 4, 5] and not t["missing_folds"] and
                len(t["folds"]) == 5 and set(t["pooled"]["models"]) == set(MODELS),
                "Task is missing a fold or a model")
    design = load(evidence / "design/DESIGN_TIMING_AUDIT.json")
    restore = load(evidence / "restoration/ALL100_RESTORATION.json")
    registry = load(evidence / "inference_package/REGISTRY.json")
    package = load(evidence / "package_audit/PACKAGE_AUDIT.json")
    require(design["planned_folds_verified"] == design["completed_saved_fold_metadata_verified"] == 100 and
            design["paired_price_return_designs_verified"] == 50, "Design audit is not complete")
    require(restore["completed_folds"] == 100 and restore["neural_models"] == 400 and
            restore["all_arrays_exact"] and restore["max_absolute_difference"] == 0 and
            restore["saved_target_arrays_opened"] is False, "Full-array restoration differs")
    require(registry["core_comparison_complete"] and registry["five_fold_tasks"] == 20 and
            package["core_comparison_complete"] and package["tasks_verified"] == 20 and
            package["models_verified"] == 160, "Package is not complete and audited")
    return receipt, results, tasks, design, restore, registry, package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-aggregation", type=Path, default=Path(__file__).resolve().parents[1] /
                        "posthoc_seed_aggregation/20261108_complete_reporting")
    args = parser.parse_args()
    evidence, parent, output = (p.resolve() for p in (args.evidence, args.parent, args.output))
    receipt, result, tasks, design, restore, registry, package = validate(evidence)
    aggregation_directory = args.seed_aggregation.resolve()
    aggregation = load(aggregation_directory / "SEED_AGGREGATION_AUDIT.json")
    require(aggregation["seeds"] == [2026, 2027, 2028] and aggregation["origins"] == 1970 and
            aggregation["new_training_started"] is False and aggregation["independent_test"] is False and
            aggregation["Q3_values_read"] is False and len(aggregation["rows"]) == 8,
            "Existing monthly seed aggregation differs")
    for name, expected in aggregation["artifacts"].items():
        require(sha(aggregation_directory / name) == expected, "Seed aggregation CSV changed")
    require(not output.exists(), "Preserve previous manuscript releases; choose a new output directory")
    original = (parent / "PRIVATE_WORKING_MANUSCRIPT_EN.md").read_text(encoding="utf8")
    text = original
    lookup = {(t["profile"], t["horizon"], t["target"]): t for t in tasks}
    prices = [lookup[p, h, "future_close"] for p, h in TASKS]
    returns = [lookup[p, h, "future_log_return"] for p, h in TASKS]
    price_wins = sum(t["pooled"]["models"][m]["relative_improvement"] > 0 for t in prices for m in NEURAL)
    return_wins = sum(t["pooled"]["models"][m]["relative_improvement"] > 0 for t in returns for m in NEURAL)
    h60 = lookup["minute", 60, "future_log_return"]
    h60_gru = h60["pooled"]["models"]["gru"]["relative_improvement"]
    h60_change = f"{abs(100 * h60_gru):.2f}% " + ("lower" if h60_gru > 0 else "higher")
    status = text.splitlines()[2]
    require(status.startswith("**WORKING MANUSCRIPT:"), "Parent draft identity differs")
    text = once(text, status + "\n\n", "")
    abstract = (
        "Forecast accuracy for one stock can vary across horizons and target representations. We compared "
        "persistence, drift, seasonal repetition, ridge regression, an MLP, GRU, LSTM and an adapted xPatch "
        "model on Apple Inc. (AAPL) prices and cumulative log returns across five temporal scales. The core "
        "comparison completed all 100 chronological development folds for twenty tasks, using one fixed "
        f"training seed. Among forty neural task–model comparisons for nominal prices, {price_wins} had lower "
        f"pooled endpoint MSE than persistence. The minute H60 return GRU had {h60_change} pooled endpoint "
        "MSE than zero return across all five folds. In a post-hoc split-neutral monthly comparison, the "
        "GRU’s relative MSE improvements were 3.96%, -24.67% and 3.99% across three seeds. The annual task "
        "had 47 overlapping validation origins. We provide complete task tables, descriptive uncertainty "
        f"estimates and a prediction interface; restoration reproduced all {restore['neural_models']} saved "
        "neural predictors exactly. Prior exposure, limited annual history, nominal split effects and "
        "finite training recipes constrain the conclusions. These historical development results provide "
        "neither independent confirmation nor an evaluation of trading performance.")
    text = section(text, "## Abstract", "## 1. Introduction", abstract)
    intro = (
        "The same forecasting method need not behave alike over a few trading minutes and over several "
        "months. Observation frequency changes the available history, the events represented by a bar and "
        "the overlap between successive targets. The target representation also matters: a small price "
        "error can reflect persistence, while cumulative-return errors describe a different objective.\n\n"
        "We studied a single asset using a common set of forecasting and evaluation rules. At each origin, "
        "the model predicted the full path of future nominal closes or cumulative log returns relative to "
        "that origin. The study addressed three questions:\n\n"
        "1. How did the eight methods compare with persistence at each frequency and horizon when they "
        "used the same forecast origins?\n"
        "2. How did direct price and cumulative-return targets compare with their respective controls, "
        "and how did nominal corporate-action discontinuities affect the reported errors?\n"
        "3. Could the saved forecasts and historical inference interface reproduce the reported results, "
        "and what did the annual-window and post-hoc seed checks reveal about their limits?\n\n"
        "All twenty core tasks completed five expanding chronological folds. The contribution is a "
        "documented comparison of these finite training recipes, with full task reporting, explicit "
        "information boundaries and restoration checks. It does not presume that a neural network beats "
        "a simple baseline or claim a new architecture. A planned shared-horizon experiment was not run; "
        "we retain its proposed matched controls as future work rather than drawing a mechanism conclusion.")
    text = section(text, "## 1. Introduction", "## 2. Related work", intro)
    text = once(text, "the present draft does not claim to implement the test", "this study did not implement the test")
    text = once(text, "A separate Alpaca daily candidate covers", "The separate Alpaca daily series covers")
    text = once(text, "For the daily-bar candidate, aggregation is defined", "For the daily bars, aggregation was defined")
    text = once(text, "this development snapshot", "this historical development study")
    text = once(text, "The final source must support this price convention.",
                "Source validation must distinguish nominal and retrospectively adjusted price conventions.")
    text = once(text, "The current exploratory runs use", "The core runs used")
    text = once(text, "The current executor uses periods", "The executor used periods")
    text = once(text, "The current independent development runs use", "The independent development runs used")
    text = once(text, "Section 5 summarizes realized updates for all completed fits in this snapshot. "
                "The accompanying training table retains missing fits and links each completed fit to its hashed learning history.",
                "Section 5 summarizes realized updates for all 400 core neural fits. The training supplement "
                "links each fit to its hashed learning history.")
    text = once(text, "At the audit snapshot, saved origins and decision/target completion timestamps matched "
                "the admitted bar positions in each of 85 completed folds.",
                "Saved origins and decision/target completion timestamps matched the admitted bar positions "
                "in all 100 completed folds.")
    text = once(text, "This comparison has not been run and contributes no empirical result to the present snapshot.",
                "This comparison was not admitted under the remaining compute budget and was not run. "
                "It is future work and contributes no empirical result to this study.")
    text = once(text, "If the core matrix and the compute admission are completed within budget, the planned "
                "mechanism comparison uses", "The unexecuted mechanism plan specified")
    # The unrun mechanism belongs with future work, outside executed methods.
    before_optional, optional_and_rest = text.split("### 4.3 Optional shared-horizon comparison", 1)
    optional_body, after_optional = optional_and_rest.split("### 4.4 Metrics and uncertainty", 1)
    text = before_optional + "### 4.3 Metrics and uncertainty" + after_optional
    text = once(text, "### 4.3 Metrics and uncertainty\n\n",
                "### 4.3 Metrics and uncertainty\n\n"
                "The initial plan designated ten principal endpoints: annual H12, quarterly H3, "
                "monthly H21, hourly H8 and minute H60, each for both targets. The remaining "
                "horizons and full-path metrics are auxiliary. All twenty tasks are reported.\n\n")
    text = once(text, "### 4.5 Reproducible prediction interface", "### 4.4 Reproducible prediction interface")
    # Retain the mathematical description of the unrun plan, clearly separated from results.
    start = "We checked the interface with fictional prices across the planned task shapes."
    end = "A further CPU check started from immutable real-data mid-epoch GRU and xPatch checkpoints."
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    inference = (
        "We checked the interface with fictional prices across the planned task shapes. The historical "
        f"full-array audit covered all {restore['neural_models']} neural predictors and "
        f"{restore['origin_model_paths']:,} origin-model prediction paths across the 100 original folds. "
        "Every restored path reproduced its saved values exactly with the original prediction batches "
        "of 512. The union reused 85 hash-matched fold checks and added fresh-process checks for the "
        "fifteen newly completed folds. Predictor restoration opened no saved target arrays. The release "
        "records the CPU runtime and artifact identities; exact equality describes these checked "
        "artifacts and batches.\n\n"
        "A new complete twenty-task inference package retained all eight methods for each task and "
        "selected the latest chronological fold, fold 5, without using model scores to choose it. Its "
        "registry recorded all 100 completed folds and all twenty five-fold tasks. A separate fresh-process "
        "API check used one original origin per task for 160 model forecast calls. Neural outputs were "
        "checked with relative tolerance 10⁻⁵ and absolute tolerance 10⁻⁶; non-neural outputs used 10⁻¹⁰ "
        "and 10⁻¹², respectively. These single-origin checks have a different numerical scope from the "
        "exact full-array restoration. The package contains model parameters and metadata; callers "
        "supply historical inputs. The selected models remain evaluation-fold models and were not "
        "refitted on the full history for production use.\n\n")
    text = head + inference + end + tail
    snapshot = (
        "The complete historical development comparison contains all 100 original task-folds and all "
        "twenty five-fold tasks. Each task includes all eight methods under the recorded recipes and "
        "seed 2026. No shared-versus-separate experiment was run. Percentages below describe endpoint "
        "MSE changes relative to matched persistence or zero return; they are not investment returns.")
    text = section(text, "## 5. Experimental results", "### Development coverage and baseline magnitudes", snapshot)
    coverage = (
        "Only complete folds using every eligible validation origin are counted. The annual task used "
        "approximately ten years of daily history and a twelve-month lookback, departing from the "
        "initial longer-history design.\n\n" +
        table(["Task", "Price folds", "Return folds"],
              [[label(lookup[p, h, "future_close"]), "5/5", "5/5"] for p, h in TASKS]) + "\n\n"
        "The following persistence errors provide the denominators for all twenty tasks. Rows pool "
        "validation origins across five folds. Win counts use unrounded pooled errors. Price and return errors use different units and are "
        "not averaged across targets. Nominal-price results include stock-split discontinuities.\n\n" +
        table(["Task", "Origins per target", "Price MSE (USD per share)²", "Log-return MSE"],
              [[label(p), f"{p['pooled']['origins']:,}", num(p['pooled']['models']['last_close']['endpoint_mse']),
                num(r['pooled']['models']['last_close']['endpoint_mse'])] for p, r in zip(prices, returns, strict=True)]))
    text = section(text, "### Development coverage and baseline magnitudes", "### Cumulative log-return target", coverage)
    def summary(items):
        return table(["Task", "Origins"] + list(NAMES[1:]),
                     [[label(t), f"{t['pooled']['origins']:,}"] +
                      [pct(t["pooled"]["models"][m]["relative_improvement"]) for m in MODELS[1:]] for t in items])
    text = section(text, "### Cumulative log-return target", "### Nominal close-price target", summary(returns))
    price_body = summary(prices) + "\n\n" + (
        f"Across all ten price tasks, {price_wins} of forty neural task–model comparisons had lower pooled "
        "endpoint MSE than persistence. The comparison concerns the recorded recipes and finite budget; "
        "it does not establish an architecture-wide limit. Small baseline errors can produce large "
        "percentage changes, so Appendix A retains the absolute errors.\n\n"
        "![Figure 2. Endpoint MSE ratios for all twenty completed tasks. Each cell is the model’s pooled "
        "endpoint MSE divided by the matched persistence or zero-return MSE; values below one favor the "
        "model. Both panels share a logarithmic color scale centered at one. Labels are rounded; "
        "Appendix A reports the underlying errors. The annual rows use the short-history variant.]"
        "(figures/completed_task_error_ratios.png){width=10.3in}")
    text = section(text, "### Nominal close-price target", "### Realized training budget", price_body)
    with (evidence / "results/TRAINING_MODELS.csv").open(newline="", encoding="utf8") as stream:
        training = list(csv.DictReader(stream))
    require(len(training) == 400 and all(r["status"] == "completed" for r in training), "Neural fit table is incomplete")
    budget_rows = []
    for scale in SCALES + ("total",):
        rows = training if scale == "total" else [r for r in training if r["profile"] == scale]
        budget_rows.append([scale.title(), len(rows), sum(r["epoch_cap_reached"] == "True" for r in rows),
                            sum(r["stopped_early"] == "True" for r in rows),
                            num(statistics.median(int(r["steps"]) for r in rows))])
    timing = result["elapsed_accounting"]
    budget = (
        "The core matrix comprised 400 completed neural model-fold fits. All used a 20-epoch cap and "
        "patience 3. The cap and early-stop flag can overlap at the final allowed epoch. Counts exclude "
        "the separately identified post-hoc anchoring and split-neutral runs.\n\n" +
        table(["Scale", "Neural fits", "At 20 epochs", "Early-stop flag", "Median updates"], budget_rows) + "\n\n"
        "A cap or stopping condition does not establish convergence. The CSV supplement retains actual "
        "epochs, selected epochs, update counts, fit/stop sample counts and learning-history hashes.\n\n"
        "The final fifteen original folds were completed under a separately approved local continuation "
        "of at most ten wall-clock hours, initially with six training processes and subsequently eight by direct user instruction, two logical CPUs each, "
        "and no GPU training. Original seeds, configurations, split identities and stopping rules were "
        "preserved. The eight-worker amendment retained the original absolute deadline. A controller-handoff protection freeze preserved all six then-running worker identities and copied 97 checkpoint/state files (294,219,271 bytes); the same processes resumed without restarting their model fits. Missing source files after a separately recorded workspace branch checkout were restored from the original hash-matched commit. Seven interrupted workers resumed from retained state, and eight original folds "
        "started afterward. No additional tuning or mechanism experiment was included.\n\n"
        f"The {timing['unique_completed_stage_records']} unique completed-stage receipts retain raw elapsed "
        f"envelopes totaling {timing['sum_recorded_wall_envelopes_seconds'] / 3600:.3f} stage-hours. Some timers "
        "include the requested night pause, the later budget freeze and a wait for an available CPU "
        "slot. A separate cost note records these limitations. The compiler’s night-only subtraction "
        "does not remove every suspension and is not reported as active computation. Concurrent stages "
        "overlap; the raw sum is neither total study wall time nor measured CPU time. Per-model and "
        "total study CPU costs were not measured comprehensively.")
    text = section(text, "### Realized training budget", "### Minute-scale results at the recorded snapshot", budget)
    fold_rows = [[f["fold"], f"{f['origins']:,}", num(f["models"]["last_close"]["endpoint_mse"])] +
                 [pct(f["models"][m]["relative_improvement"]) for m in NEURAL] for f in h60["folds"]]
    gru_folds = [f["models"]["gru"]["relative_improvement"] for f in h60["folds"]]
    minute = (
        "The minute matrix completed all forty folds for H5, H15, H30 and H60, separately for prices "
        "and cumulative returns. All five folds enter each row of the main summary. Different horizons "
        "retain their own eligible origins; matching fold numbers does not make their origin sets identical.\n\n"
        f"For H60 returns, all five folds contributed {h60['pooled']['origins']:,} origins. The GRU pooled "
        f"endpoint MSE was {h60_change} than zero return, with positive relative improvement in "
        f"{sum(v > 0 for v in gru_folds)} of five folds. The table retains every fold, including unfavorable "
        "results; the earlier sampled-origin pilot is excluded.\n\n" +
        table(["H60 return fold", "Origins", "Zero-return MSE", "MLP", "GRU", "LSTM", "xPatch"], fold_rows))
    text = section(text, "### Minute-scale results at the recorded snapshot", "### Annual sensitivity: nonoverlapping forecast windows", minute)
    text = once(text, "### Minute-scale results at the recorded snapshot", "### Minute-scale results and fold variation")
    old_uncertainty = text.split("### Descriptive uncertainty across completed tasks\n\n", 1)[1].split("\n\nFor example,", 1)[0]
    uncertainty = (
        "We summarized paired endpoint squared-loss differences for all twenty tasks using 5,000 "
        "non-circular moving-block resamples, seed 2026 and nominal 95% percentile intervals. The full "
        "source session grid retained gaps between observed validation groups. The supplement contains "
        f"320 task–model–block rows, including {result['interval_status_counts'].get('descriptive_development_only', 0)} "
        f"varying descriptive intervals and {result['interval_status_counts'].get('degenerate_bootstrap', 0)} "
        "degenerate cases with undefined bounds. Each task uses all five folds. Negative model-minus-baseline "
        "loss differences favor the model. We report no development p-values; prior score exposure, "
        "dependence and nonstationarity limit interpretation.")
    text = once(text, old_uncertainty, uncertainty)
    aggregation_lookup = {r["model"]: r for r in aggregation["rows"]}
    seed_summary = (
        "### Seed averages and ensemble errors\n\n"
        "For the separate post-hoc split-neutral monthly diagnostic, we also distinguished the "
        "arithmetic mean of three seed-specific endpoint MSEs from the error of the arithmetic mean "
        "prediction. The latter averaged predictions with one-third weight per seed before calculating "
        "the error, using the same 1,970 origins and five folds. No ensemble weights were fitted. "
        "The diagnostic series multiplied each raw close by cumulative split ratios effective by that "
        "bar, so future actions did not rewrite earlier bar values. The table retains all eight methods, "
        "including the deterministic controls.\n\n" +
        table(["Model", "Mean seed endpoint MSE", "Mean seed improvement", "Ensemble endpoint MSE", "Ensemble improvement"],
              [[name, num(aggregation_lookup[m]["mean_seed_endpoint_mse"]), pct(aggregation_lookup[m]["mean_seed_relative_improvement"]),
                num(aggregation_lookup[m]["ensemble_endpoint_mse"]), pct(aggregation_lookup[m]["ensemble_relative_improvement"])]
               for m, name in zip(MODELS, NAMES, strict=True)]) + "\n\n"
        "Averaging predictions can reduce squared error even when the mean individual loss is worse "
        "than baseline, so the two summaries are not interchangeable. These calculations reuse the "
        "same exposed periods and were reported after the seed results were known. No ensemble "
        "uncertainty interval or independent confirmation test was added; the result does not establish "
        "a stable advantage for any individual seed or architecture.\n\n")
    text = once(text, "### Post-hoc price anchoring", seed_summary + "### Post-hoc price anchoring")
    conclusion = (
        "The study completed twenty tasks, eight methods and five chronological folds per task under "
        "one core training seed. It concerns one stock and uses different source histories at different "
        "frequencies. The annual analysis contains 47 overlapping origins and only four selected "
        "nonoverlapping outcome windows, which can remain dependent. Retrospective vendor corrections "
        "and incomplete evidence of historical point-in-time availability limit causal backtest claims.\n\n"
        f"Under the recorded recipes, {price_wins} of forty neural price comparisons and {return_wins} of "
        "forty neural return comparisons had lower pooled endpoint MSE than their matched baseline. "
        f"For minute H60 returns, the GRU error was {h60_change} than zero return. These are descriptive "
        "comparisons of the finite training recipes. They do not establish an architecture’s attainable "
        "limit, and a small positive pooled difference does not establish stable predictive value.\n\n"
        "Target construction materially affected the errors. One split event contributed most of the "
        "nominal monthly baseline’s squared error, while the post-hoc split-neutral GRU changed sign "
        "across seeds. The main matrix has one seed per task; the three-seed check applies only to the "
        "separate monthly diagnostic. A shared-horizon mechanism comparison was not run.\n\n"
        "The complete tables and restoration checks make the reported historical forecasts auditable. "
        "They do not remove prior exposure or supply independent validation. The protected third-quarter "
        "data remained sealed. Independent confirmation requires a separately frozen protocol and "
        "eligible future data, with enough time to realize each horizon. No trading strategy, transaction "
        "cost or profitability analysis was performed, and forecast-error changes are not investment returns.")
    text = section(text, "## 6. Limitations and conclusion", "## 7. Data and code availability", conclusion)
    availability = (
        "The repository contains task definitions, source and split checks, independent scoring, "
        "historical prediction interfaces, full-matrix reporting and bounded checkpoint restart checks. "
        "The release receipt identifies the code commit and remote verification. The versioned source "
        "manifest identifies the Alpaca daily and minute files and derived bars. Permission for Alpaca "
        "and Kibot data was confirmed by the researcher; the reported experiments used Alpaca.\n\n"
        "The local numerical supplement contains 160 pooled model rows, 800 completed model-fold rows, "
        "2,544 model/step rows, 320 interval rows and 400 completed core neural-fit records. Additional "
        "CSVs retain the separate monthly seed and ensemble summaries. The complete "
        "inference package provides twenty task interfaces and eight methods per task, together with its "
        "registry and fresh-process audit. Source data, weights, prediction arrays, the manuscript and "
        "machine receipts remain local artifacts rather than repository source files. Reproducing the "
        "numerical results requires the matching source versions and artifacts. No public data archive "
        "or DOI has been assigned to this local release.")
    text = section(text, "## 7. Data and code availability", "## References", availability)
    editorial = ("Reference metadata, versions, and the scope of prior reading must be checked again against the target venue "
                 "before submission. The issue number for reference 17 is omitted pending verification. References 13 and 17 "
                 "do not imply that their associated procedures have already been run.\n\n")
    text = once(text, editorial, "")
    appendix = (
        "All twenty tasks completed folds 1–5 with all eight methods. Endpoint MSE, RMSE and MAE, and "
        "path errors are pooled by origin count. Price MSE uses (USD per share)²; price RMSE and MAE "
        "use USD per share. Return MSE uses squared log-return units. Improvement is 100 × (1 − model "
        "endpoint MSE / matched persistence MSE). Direction accuracy covers nonzero endpoint forecasts; "
        "an all-abstaining predictor has no accuracy value. The CSV supplement retains every fold, "
        "path MAE and stepwise errors. Recorded New York completion times do not establish historical "
        "vendor delivery or revision times.\n\n")
    for t in tasks:
        p, folds = t["pooled"], t["folds"]
        appendix += "### " + label(t) + (": nominal close" if t["target"] == "future_close" else ": cumulative log return") + "\n\n"
        appendix += (f"Completed folds: 1, 2, 3, 4, 5; no missing folds; {p['origins']:,} origins. "
                     f"First and last decisions: {folds[0]['first_decision_ny']} to {folds[-1]['last_decision_ny']}; "
                     f"last target completion: {max(f['last_completion_ny'] for f in folds)}.\n\n")
        appendix += table(["Model", "End MSE", "End RMSE", "End MAE", "Path MSE", "Improvement", "Direction coverage", "Active accuracy"],
                          [[name] + [num(p["models"][model][k]) for k in ("endpoint_mse", "endpoint_rmse", "endpoint_mae", "path_mse")] +
                           [pct(p["models"][model][k]) for k in ("relative_improvement", "direction_coverage", "direction_accuracy_when_active")]
                           for model, name in zip(MODELS, NAMES, strict=True)]) + "\n\n"
    text = section(text, "## Appendix A. Detailed historical development results", "## Appendix B. Forecast timing and original fold design", appendix)
    timing_rows = []
    for p, h in TASKS:
        d = next(t for t in design["tasks"] if (t["profile"], t["horizon"], t["target"]) == (p, h, "future_close"))
        a = d["planned_origin_timing"]
        require(a == d["completed_origin_timing"], "Completed timing differs from the original design")
        timing_rows.append([p.title() + f" H{h}", f"{a['origins']:,}"] +
                           [num(a[k]) for k in ("elapsed_hours_min", "elapsed_hours_median", "elapsed_hours_p95", "elapsed_hours_max")] +
                           [pct(a["cross_session_fraction"])] )
    text = text.split("## Appendix B. Forecast timing and original fold design", 1)[0] + (
        "## Appendix B. Forecast timing and original fold design\n\n"
        "All five original validation folds of each task were completed and their saved timing metadata "
        "verified. Price and return tasks share the same design, so each horizon appears once. Origin "
        "counts agree with the completed results but do not represent independent observations.\n\n" +
        table(["Task", "Origins per target", "Minimum h", "Median h", "95th percentile h", "Maximum h", "Cross-session paths"], timing_rows) + "\n\n"
        "Durations are clock hours between recorded decision and endpoint completion times. Percentiles "
        "use linear interpolation over origins. Daily and monthly completions follow the conservative "
        "next-New-York-midnight convention in Section 3; daylight-saving changes enter the elapsed "
        "durations. A cross-session path ends in a different session from its origin.\n\n"
        "The supplement retains all 100 original fold designs, 400 partition hashes, fit/stop/train/validation "
        "counts, strict label-purge gaps, decision and endpoint ranges, and all 100 completed saved-file "
        "metadata checks. The audit changed no outcome, configuration, original split identity or protected-period status.\n")
    text += "\n## Appendix C. Unexecuted shared-horizon plan\n\n" + optional_body.strip() + "\n"
    # Retain every mathematical expression from the audited scientific source.
    math_pattern = r"\\\[.*?\\\]|\\\(.*?\\\)"
    require(Counter(re.findall(math_pattern, original, re.S)) == Counter(re.findall(math_pattern, text, re.S)),
            "A scientific equation changed")
    refs = lambda s: s.split("## References\n", 1)[1].split("## Appendix A.", 1)[0].strip()
    require(refs(original).replace(editorial.strip(), "").strip() == refs(text), "Bibliography changed")
    forbidden = ("85/100", "85 of 100", "71-fold snapshot", "minute matrix remains incomplete", "minute matrix remained incomplete",
                 "missing folds: 4", "WORKING MANUSCRIPT", "PRIVATE WORKING DRAFT", "unfinished minute")
    require(not any(s.lower() in text.lower() for s in forbidden), "Stale partial-study statement remains")
    output.mkdir(parents=True)
    (output / "MANUSCRIPT_EN.md").write_text(text, encoding="utf8")
    for name in ("paper_artifacts", "reference_evidence_20261108", "monthly_split_groups"):
        shutil.copytree(parent / name, output / name)
    for name in ("DATA_PERMISSION_CONFIRMATION.json", "DATA_PROVENANCE_MANIFEST.json", "REAL_TRAINING_RESTART_AUDIT.json",
                 "ACTUAL_NIGHT_RESUME_VERIFICATION.json", "PRIVATE_ANNUAL_NONOVERLAP_RESULTS.csv"):
        source = parent / name
        if source.exists():
            shutil.copy2(source, output / name)
    shutil.copytree(evidence / "results", output / "all_task_results")
    shutil.copytree(evidence / "design", output / "forecast_timing")
    shutil.copytree(aggregation_directory, output / "posthoc_seed_aggregation")
    for source, dest in ((evidence / "ALL100_EVIDENCE_RECEIPT.json", "ALL100_EVIDENCE_RECEIPT.json"),
                         (evidence / "restoration/ALL100_RESTORATION.json", "ALL100_RESTORATION.json"),
                         (evidence / "package_audit/PACKAGE_AUDIT.json", "STUDY_PACKAGE_AUDIT.json")):
        shutil.copy2(source, output / dest)
    audit = {"status": "complete_core_manuscript_built_pending_document_and_visual_checks",
             "created_at_utc": datetime.now(timezone.utc).isoformat(), "parent": str(parent),
             "parent_markdown_sha256": sha(parent / "PRIVATE_WORKING_MANUSCRIPT_EN.md"),
             "evidence": str(evidence), "evidence_receipt_sha256": sha(evidence / "ALL100_EVIDENCE_RECEIPT.json"),
             "markdown_sha256": sha(output / "MANUSCRIPT_EN.md"), "script_sha256": sha(__file__),
             "completed_folds": 100, "complete_tasks": 20, "core_training_seed": 2026,
             "neural_fits": 400, "neural_price_comparisons_below_baseline": price_wins,
             "neural_return_comparisons_below_baseline": return_wins,
             "h60_return_gru_relative_improvement": h60_gru, "equations_preserved": True,
             "posthoc_seed_aggregation_sha256": sha(aggregation_directory / "SEED_AGGREGATION_AUDIT.json"),
             "unexecuted_mechanism_moved_to_appendix_C": True,
             "references_preserved": True, "mechanism_run": False, "independent_test": False,
             "Q3_values_read": False, "source_receipts_rewritten": False,
             "cpu_cost_limitation": "Raw envelopes and night-only subtraction are not actual active computation."}
    (output / "MANUSCRIPT_BUILD_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("status", "completed_folds", "complete_tasks",
                                          "neural_price_comparisons_below_baseline", "h60_return_gru_relative_improvement")}))


if __name__ == "__main__":
    main()
