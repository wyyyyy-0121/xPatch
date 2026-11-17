"""Prepare a compact English paper from checked historical development evidence.

Partial evidence is admitted only with --layout-preview and remains explicitly
labelled throughout. A complete paper requires the all100 gate from the long
report. Author names and emails must come from the user, never Git metadata.
This utility performs no training, market-data retrieval or holdout scoring.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from textwrap import dedent

from build_complete_manuscript import validate as validate_all100

TOOLS = Path(__file__).resolve().parent
BASE = TOOLS.parent
MODELS = ("last_close", "drift", "seasonal", "ridge", "mlp", "gru", "lstm", "xpatch")
MODEL_NAMES = ("Persistence", "Drift", "Seasonal", "Ridge", "MLP", "GRU", "LSTM", "xPatch")
TASKS = (("annual", 12), ("quarterly", 3), ("monthly", 21),
         ("hourly", 1), ("hourly", 4), ("hourly", 8),
         ("minute", 5), ("minute", 15), ("minute", 30), ("minute", 60))
TARGETS = ("future_close", "future_log_return")
PARENT = BASE / "private_working_manuscripts/20261107T211544188288Z/PRIVATE_WORKING_MANUSCRIPT_EN.md"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def pct(value):
    return f"{100 * value:.2f}%"


def label(profile, horizon):
    return profile.title() + f" H{horizon}"


def author_information(path):
    info = load(path)
    affiliations = {a["id"]: a for a in info["affiliations"]}
    check(len(affiliations) == len(info["affiliations"]), "Duplicate affiliation ids")
    check(affiliations["1"]["school"] == "High School Affiliated to Shanghai Jiao Tong University" and
          affiliations["1"]["city"] == "Shanghai" and affiliations["1"]["postal_code"] == "200439" and
          affiliations["1"]["country"] == "China", "Recheck official affiliation evidence before changing it")
    authors = info["authors"]
    ready = bool(authors) and info["status"] in ("confirmed_by_user", "confirmed_by_user_email_self_fill")
    if ready:
        for a in authors:
            check(bool(a.get("name_en", "").strip()) and len(a["name_en"].split()) >= 2,
                  "User-supplied full English byline is required")
            check(a.get("affiliation_id") in affiliations, "Unconfirmed author affiliation")
            email = a.get("email")
            if a.get("email_completion") == "author_will_fill":
                check(info["status"] == "confirmed_by_user_email_self_fill" and email is None,
                      "An email placeholder requires the user's explicit self-fill choice")
            else:
                check(isinstance(email, str) and re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email) is not None and
                      re.search(r"[A-Za-z]", email.split("@")[0]) is not None,
                      "Use the user's email with letters in its local part")
    # The screenshot's example identity is never imported as an author.
    byline = ", ".join(a["name_en"] + "^" + a["affiliation_id"] + "^" for a in authors) if ready else "[Full author name pending]"
    lines = [byline]
    used = {a["affiliation_id"] for a in authors} if ready else {"1"}
    for key, a in affiliations.items():
        if key not in used:
            continue
        unit = ([a["department"]] if a.get("department") else []) + [a["school"]]
        lines.append("^" + key + "^ " + ", ".join(unit))
        lines.append(", ".join([a["city"], a["postal_code"], a["country"]]))
    lines.append("; ".join(a["email"] if a.get("email") else "[Email to be added by the author]"
                           for a in authors) if ready else "[Author email pending]")
    return info, ready, "\n\n".join(lines)


def references():
    text = PARENT.read_text(encoding="utf8").split("## References\n\n", 1)[1].split("\n## ", 1)[0]
    lookup = {}
    for line in text.splitlines():
        match = re.match(r"(\d+)\. (.+)", line)
        if match:
            lookup[int(match[1])] = match[2]
    ids = (1, 2, 3, 4, 5, 6, 8, 9, 12, 14, 15, 16)
    check(all(i in lookup for i in ids), "Checked reference inventory changed")
    return "\n\n".join(f"[{n}] {lookup[i]}" for n, i in enumerate(ids, 1)), ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--supplement", type=Path)
    parser.add_argument("--authors", type=Path, default=TOOLS / "AUTHOR_INFORMATION.json")
    parser.add_argument("--layout-preview", action="store_true")
    args = parser.parse_args()
    source = args.results.resolve()
    if source.is_dir():
        source /= "RESULTS.json"
    results = load(source)
    check(results["independent_test"] is False and results["p_values_reported"] is False and
          results["Q3_values_read"] is False, "Historical evidential scope changed")
    for name, digest in results["csv_sha256"].items():
        check(sha(source.parent / name) == digest, "Result CSV changed: " + name)
    complete = results["completed_folds"] == 100 and results["complete_tasks"] == 20
    restoration = None
    if not args.layout_preview:
        check(complete and args.evidence is not None and args.supplement is not None,
              "A final paper requires all100 evidence and its full supplementary report")
        evidence = args.evidence.resolve()
        verified = validate_all100(evidence)
        restoration = verified[4]
        check(sha(source) == sha(evidence / "results/RESULTS.json"), "Paper results differ from verified release")
        supplement = args.supplement.resolve()
        check((supplement / "DOCUMENT_CONTENT_VERIFICATION.json").exists() and
              (supplement / "MANUSCRIPT_EN.pdf").exists(), "Supplement document is not generated and checked")
    else:
        check(not complete, "Use the full evidence gate for completed results")
    tasks = {(t["profile"], t["horizon"], t["target"]): t for t in results["tasks"]}
    check(len(tasks) == 20 and {(p, h) for p, h, _ in tasks} == set(TASKS), "Missing or duplicate tasks")
    check(set(t for _, _, t in tasks) == set(TARGETS), "Target set differs")
    if complete:
        check(all(t["completed_folds"] == [1, 2, 3, 4, 5] and not t["missing_folds"] for t in tasks.values()),
              "Five-fold comparisons are not complete")
    for task in tasks.values():
        check(set(task["pooled"]["models"]) == set(MODELS), "Missing method")
        base = task["pooled"]["models"]["last_close"]["endpoint_mse"]
        check(math.isfinite(base) and base > 0, "Undefined persistence denominator")
        for name, metrics in task["pooled"]["models"].items():
            check(math.isclose(metrics["relative_improvement"], 1-metrics["endpoint_mse"]/base,
                               abs_tol=1e-12), "Relative improvement differs")
    authors, authors_ready, author_block = author_information(args.authors)
    eligible = [t for t in tasks.values() if t["scope"] == "complete_five_fold_development"]
    price = [t for t in eligible if t["target"] == "future_close"]
    returns = [t for t in eligible if t["target"] == "future_log_return"]
    wins = lambda ts: sum(t["pooled"]["models"][m]["relative_improvement"] > 0 for t in ts for m in MODELS[4:])
    price_wins, return_wins = wins(price), wins(returns)
    nprice, nreturn = 4*len(price), 4*len(returns)
    annual = tasks["annual", 12, "future_log_return"]["pooled"]["models"]
    quarterly = tasks["quarterly", 3, "future_log_return"]["pooled"]["models"]
    minute60 = tasks["minute", 60, "future_log_return"]
    h60_models = minute60["pooled"]["models"]
    h60_gru = h60_models["gru"]["relative_improvement"]
    h60_word = "lower" if h60_gru >= 0 else "higher"
    minute_scope = "all five folds" if complete else f"the {len(minute60['completed_folds'])} available folds"
    seed_dir = BASE / "posthoc_seed_aggregation/20261108_complete_reporting"
    with (seed_dir / "SEED_VERSUS_ENSEMBLE.csv").open(newline="") as stream:
        seed_rows = list(csv.DictReader(stream))
    check(tuple(r["model"] for r in seed_rows) == MODELS and
          all(r["origins"] == "1970" and r["seeds"] == "3" for r in seed_rows),
          "Post-hoc seed aggregation scope changed")
    seed_check = load(seed_dir / "SEED_AGGREGATION_CROSSCHECK.json")
    check(seed_check["new_training_started"] is False and seed_check["Q3_values_read"] is False and
          max(r["variance_identity_absolute_error"] for r in seed_check["rows"]) < 1e-15,
          "Seed aggregation crosscheck failed")
    seed_table = table(["Method", "Mean seed loss", "Prediction mean"],
                       [[MODEL_NAMES[i], pct(float(r["mean_seed_relative_improvement"])),
                         pct(float(r["ensemble_relative_improvement"]))] for i, r in enumerate(seed_rows)])
    baseline_rows, baseline_fields, cells = [], [], []
    for p, h in TASKS:
        a, b = tasks[p, h, TARGETS[0]], tasks[p, h, TARGETS[1]]
        na, nb = a["pooled"]["origins"], b["pooled"]["origins"]
        counts = f"{na:,}" if na == nb else f"{na:,}/{nb:,}"
        ca, cb = len(a["completed_folds"]), len(b["completed_folds"])
        folds = "" if complete else f" ({ca}/{cb})"
        mse_p = a["pooled"]["models"]["last_close"]["endpoint_mse"]
        mse_r = b["pooled"]["models"]["last_close"]["endpoint_mse"]
        baseline_rows.append([label(p, h) + folds, counts, f"{mse_p:.4g}", f"{mse_r:.4g}"])
        baseline_fields.append({"profile": p, "horizon": h, "price_origins": na, "return_origins": nb,
                                "price_mse": mse_p, "return_mse": mse_r,
                                "price_display": f"{mse_p:.4g}", "return_display": f"{mse_r:.4g}"})
        for target in TARGETS:
            t = tasks[p, h, target]
            baseline = t["pooled"]["models"]["last_close"]["endpoint_mse"]
            for model in MODELS[1:]:
                cells.append({"profile": p, "horizon": h, "target": target, "model": model,
                              "origins": t["pooled"]["origins"],
                              "completed_folds": len(t["completed_folds"]),
                              "ratio": t["pooled"]["models"][model]["endpoint_mse"]/baseline})
    baseline_table = table(["Task", "Origins", "Price MSE", "Return MSE"], baseline_rows)
    refs, reference_ids = references()
    status = (f"LAYOUT PREVIEW — {results['completed_folds']}/100 folds; experiments incomplete." if args.layout_preview
              else "Historical development study — 100/100 folds; no independent confirmation.")
    if not authors_ready:
        status += " Author name and email pending."
    coverage_sentence = ("All twenty tasks have five completed folds (100 task-folds and 400 neural fits)." if complete else
                         f"This layout preview reports {results['completed_folds']} of 100 planned task-folds; "
                         f"{results['complete_tasks']} of twenty tasks are complete. Minute comparisons are partial.")
    comparison_scope = "complete price tasks" if not complete else "nominal-price tasks"
    restoration_text = ("All 400 neural predictor-fold outputs reproduced their saved validation arrays exactly "
                        "using the original prediction batches of 512. The predictor restoration did not open "
                        "saved target arrays. A separate complete inference-package check covered all twenty tasks "
                        "and 160 model forecast calls." if complete else
                        "The checked 85-fold restoration covers 340 neural predictors. The separately recorded "
                        "inference package is an earlier partial snapshot and is not a complete study release. "
                        "The pending full-evidence checks must finish before a complete package is claimed.")
    abstract_scope = ("The comparison comprises twenty tasks and five expanding chronological folds per task." if complete else
                      f"This incomplete development snapshot contains {results['completed_folds']} of 100 planned folds; "
                      f"only {results['complete_tasks']} tasks have five folds.")
    baseline_note = ("Every row pools five folds; counts are shared by the two targets." if complete else
                     "Parentheses show completed price/return fold counts. Unequal origin counts are shown as price/return. "
                     "Rows cover different fold sets and are not matched across tasks.")
    paper = f"# Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales\n\n{author_block}\n\n{status}\n\n"
    paper += dedent(fr"""
    ## Abstract

    Forecast comparisons for a single stock depend on the forecast horizon, target definition and persistence reference. We compare persistence, drift, seasonal repetition, ridge regression, MLP, GRU, LSTM and an adapted xPatch model on Apple Inc. (AAPL) nominal-close and cumulative-log-return paths at five temporal scales. {abstract_scope} Among {nprice} neural comparisons on {comparison_scope}, {price_wins} have lower pooled endpoint mean squared error (MSE) than persistence. For minute H60 returns, the GRU endpoint MSE is {abs(100*h60_gru):.2f}% {h60_word} than zero return across {minute_scope}. Monthly results are sensitive to the 2020 stock split: 21 overlapping paths around one event account for 74.01% of nominal zero-return squared error. A separate post-hoc split-neutral GRU comparison changes sign across three seeds. The annual task has only 47 overlapping origins. The study provides matched-origin comparisons and reproducibility checks for exposed historical development periods; it establishes neither independent predictive improvement nor trading profitability.

    **Keywords:** financial forecasting; multi-horizon evaluation; persistence; target representation; corporate actions.

    ## 1. Introduction

    Forecasting over five trading minutes and forecasting over twelve months involve different sample sizes, dependence structures and market events. A method that performs well on one target need not retain its ranking when the bar frequency or target representation changes. Price-level predictions are especially sensitive to the persistence reference: a small dollar error can coexist with an error above that of simply retaining the latest close. Cumulative returns remove the level from the outcome definition but introduce their own conventions, including the handling of corporate actions.

    We compare eight methods under a common origin, history and label design for one stock. Ten horizons, each evaluated as a nominal-price path and a cumulative-log-return path, define twenty tasks. The questions are: how do the recorded training recipes compare with matched persistence controls across these tasks, and how do target definitions and overlapping outcomes affect the interpretation? The contribution is a documented empirical comparison with explicit limits on historical evidence, rather than a new forecasting architecture. A shared-horizon mechanism was planned but was not run; it supplies no result in this paper.

    The comparison retains unfavorable results and separates the core single-seed matrix from later diagnostics. Annual nonoverlapping-window analysis, monthly split-neutral retraining, seed aggregation and residual price anchoring address specific sensitivities. They reuse already inspected market periods and cannot convert those periods into independent confirmation. The full supplementary report retains the twenty task tables, fold details, uncertainties, source identities and model-restoration records.

    ## 2. Related work

    xPatch combines exponential decomposition with separate processing of trend and residual components [1]. PatchTST uses temporal patches and channel-independent processing [2], while Temporal Fusion Transformers provide an established multi-horizon design with learned selection and gating [3]. We evaluate an adapted xPatch recipe; neither a multi-step output nor an architectural family by itself is a novelty claim.

    Financial forecasting studies differ in information sets and statistical objectives [4]. DeepLOB uses limit-order-book information for classification [5]. Ballarin et al. study multi-horizon intraday returns across a broader stock universe [6]. Kronos is pretrained on multivariate financial candlesticks [7]. These settings provide context but are not like-for-like model comparisons with our close-derived path regression. Their published scores cannot be substituted for errors on our forecast origins.

    Kamalov et al. compare price and return input features for next-step direction classification [8]; our distinction concerns future target paths. Mattera et al. include Apple in multi-horizon stock-price forecasting using rolling ARIMA, random-walk benchmarks, clustering and cross-series forecast reconciliation [9]. That work is direct precedent for multi-horizon evaluation involving AAPL. Our single-stock information set and path errors differ from theirs. Chronological evaluation must also respect the relation between fitting and assessment periods [10], and block resampling requires care with dependence and stationarity [11]. These considerations motivate the design and the restricted interpretation below.

    ## 3. Data and forecasting tasks

    ### 3.1 Source, calendar and provenance

    Experiments use Alpaca SIP/raw data. The daily source contains 2,637 rows from January 2016 through June 2026; conservative completion rules retain 2,636 daily dates and 125 complete months before the cutoff. The minute source starts in January 2021 and contains 1,061,536 raw rows. The regular-session analysis grid has all 535,620 expected minutes observed. Aggregation yields 8,238 complete hourly bars; 41,340 residual minutes that cannot form a complete hour are excluded. The XNYS calendar is a grid proxy, and a minute-derived session close need not equal the official auction close.

    A daily bar is conservatively assigned completion at the following New York midnight. This convention follows the aggregation context [12], but it does not prove when a historically final vendor value became available. The June 30, 2026 daily slot and June monthly slot are excluded because their assigned completion reaches the protected boundary. No missing bar is skipped to shorten a forecast path. Exact source versions, derived arrays and calendar checks are identified by hashes in the supplement.

    The researcher confirmed permission for all Alpaca and Kibot data on November 8, 2026. That is recorded as researcher-confirmed permission, without independent inspection of a supplier contract. Kibot contributes none of the reported results. The annual variant uses approximately ten years of history and a twelve-month lookback, accepted as a departure from the original longer-history plan. The core recipes and chronological partitions are otherwise retained.

    ### 3.2 Targets and temporal interpretation

    Let \(P_t>0\) be the close of a completed bar and \(H\) the number of subsequent expected bar positions. At every eligible origin, each target is fitted separately for \(j=1,\ldots,H\):

    \[
    Y^P_{{t,j}}=P_{{t+j}},\qquad Y^R_{{t,j}}=\log(P_{{t+j}}/P_t).
    \]

    The annual and quarterly tasks use monthly bars with \(H=12\) and \(H=3\); the monthly task uses daily bars with \(H=21\). Hourly tasks use \(H=1,4,8\), and minute tasks use \(H=5,15,30,60\). Their lookbacks are respectively 12, 36, 252, 96 and 480 bars. Labels require all expected future positions. These ten horizons and two representations define the twenty tasks.

    Minute paths remain within one regular session, so their endpoint delays are exactly \(H\) minutes. Complete hourly bars begin at the regular open and may cross sessions. Median elapsed endpoint delays are 22 clock hours for H4 and 26 for H8; every H8 path crosses a session boundary, and the longest delay is 119 hours. Thus an H8 trading-bar task cannot be described as an eight-clock-hour forecast.

    Historical inputs express past closes in the share units known at the origin, using splits already effective by then. A future split is not supplied as a feature and can create a nominal target discontinuity. Targets exclude cash dividends and are not total returns. The post-hoc split-neutral diagnostic uses a different target; it is reported separately rather than silently replacing the original outcomes.

    ## 4. Methods

    ### 4.1 Matched inputs and model recipes

    Every trained method receives a close-derived history and the current close as an anchor. If \(\widetilde P_{{u\mid t}}\) is a historical close in share units known at origin \(t\), its window is

    \[
    x_{{t,k}}=\log(\widetilde P_{{t-L+k\mid t}}/P_t),\quad k=1,\ldots,L.
    \]

    The price persistence path is \(P_t\) at every step; the return persistence path is zero. Drift extrapolates the first-to-last history change in the relevant price or log units. Seasonal repetition uses 12 monthly, 252 daily, 6 hourly or 390 minute bars; a complete observed cycle is required. Seasonal phase is defined in trading bars, so early closes can affect it. Ridge uses the same inputs, an unpenalized intercept and penalties 0.1, 1 and 10 selected by inner-stop endpoint MSE; a tie selects the smaller penalty.

    The MLP has one 64-unit ReLU hidden layer and dropout 0.1. GRU and LSTM each have one 32-unit recurrent layer. The xPatch adapter uses a causal exponential moving average with coefficient 0.1, its native dual-stream backbone and a learned anchor readout. Patch length is at most eight bars, with half-length stride bounded below by one. The hourly H1 adaptation computes a two-step head and retains the first output. This is a task-specific recipe, not a reproduction of the original xPatch training study [1].

    Means and scales for window elements, anchors and each target step are fitted only on the inner fitting partition. Standard deviations at or below \(10^{{-12}}\) are replaced by one. Neural training minimizes standardized path MSE; inner stopping selects the best unstandardized endpoint MSE. Adam uses learning rate 0.001 and batch size 128, with a cap of 20 epochs and patience 3. The core seed is 2026. Selected models are not refitted on the stopping partition. Neither an epoch cap nor early stopping proves convergence.

    ### 4.2 Chronology, scoring and descriptive uncertainty

    Each task uses five expanding chronological folds. Every training label finishes strictly before the first outer evaluation decision; inner fitting labels likewise finish before inner stopping decisions. All eight methods share the eligible origins within a task. Price and return designs are paired where their completed fold sets match. These rules address label timing but cannot remove historical vendor revisions or prior investigator exposure.

    Principal horizons in the initial plan were annual H12, quarterly H3, monthly H21, hourly H8 and minute H60, for both targets; the remaining horizons and full-path metrics are auxiliary. All twenty tasks are reported.

    For \(N\) origins, the primary error is

    \[
    \mathrm{{MSE}}_H=\frac{{1}}{{N}}\sum_{{t=1}}^N(\widehat Y_{{t,H}}-Y_{{t,H}})^2.
    \]

    Full-path MSE averages over both origins and steps. Endpoint MAE, RMSE, stepwise errors and direction accuracy with nonzero-prediction coverage are retained in the supplement. Exactly zero predicted change is an abstention; an active prediction against exactly zero realized change is incorrect. Losses pool origins across folds rather than averaging fold MSEs with equal weights. Price MSE has units of squared dollars per share; cumulative-log-return MSE is dimensionless. These losses are never averaged across representations.

    We report the model-to-persistence endpoint MSE ratio \(r=\mathrm{{MSE}}_{{\mathrm{{model}}}}/\mathrm{{MSE}}_{{\mathrm{{persistence}}}}\) and relative improvement \(100(1-r)\%\), using identical origins. A positive improvement describes lower forecast error, not investment return. Ratios are undefined for a zero baseline error; none of the reported pooled baselines is zero.

    Paired squared-loss differences use 5,000 non-circular moving-block resamples, seed 2026 and nominal 95% percentile intervals [11]. Primary/sensitivity block lengths are 12/24 monthly groups for annual tasks, 3/6 for quarterly, 21/42 daily groups for monthly, 2/5 sessions for hourly and 1/5 sessions for minute tasks. The full session grid retains gaps between validation groups. Intervals were examined after development scores were available and remain descriptive. No development p-values or independent significance claims are reported. Seed repetitions do not add independent market periods.

    ## 5. Results

    ### 5.1 Core matrix and baseline magnitudes

    {coverage_sentence} The core matrix uses one seed and the frozen recipes. Table 1 gives persistence denominators; Figure 1 retains all seven alternatives for both targets at every horizon. {baseline_note} The exhaustive model, fold and step errors remain in the supplementary report.

    Across {len(price)} complete price tasks, {price_wins} of {nprice} neural task–method comparisons have lower pooled endpoint MSE than persistence. Across {len(returns)} complete return tasks, the corresponding count is {return_wins} of {nreturn}. These are descriptive signs on the specified recipes, not independent discoveries or estimates of architecture-wide potential. Figure 1 shows why both absolute denominators and ratios are needed: a small persistence error can make modest level errors produce very large ratios.

    **Table 1.** Persistence endpoint MSE and validation-origin counts. Price units are \((\mathrm{{USD/share}})^2\); return MSE is dimensionless.

    {baseline_table}

    ![Figure 1. Endpoint MSE ratios for all twenty tasks. Each cell divides model error by matched persistence or zero-return error; ratios below one favor the model. Panels share a logarithmic color scale centered at one. Values are rounded; the supplementary CSV retains precision and absolute errors. Annual rows use the accepted short-history variant.{' Minute rows are incomplete; their fold counts are shown in Table 1.' if not complete else ' Every task pools five folds.'}](figures/short_paper_error_ratios.png){{width=7.0in}}

    The annual-return GRU improvement is {pct(annual['gru']['relative_improvement'])}, but only 47 overlapping validation origins are available. Its fold-specific improvement ranges from -234.36% in fold 1 to 88.74% in fold 2. Quarterly return MLP improvement is {pct(quarterly['mlp']['relative_improvement'])} on 52 overlapping origins. These windows provide far fewer distinct market episodes than their nominal origin counts suggest.

    For H60 returns, the GRU pooled endpoint MSE is {abs(100*h60_gru):.2f}% {h60_word} than the zero-return reference across {minute_scope}. The neural improvements for this task are {pct(h60_models['mlp']['relative_improvement'])} (MLP), {pct(h60_gru)} (GRU), {pct(h60_models['lstm']['relative_improvement'])} (LSTM) and {pct(h60_models['xpatch']['relative_improvement'])} (xPatch). Near-zero relative differences are small forecast-loss differences, with no transaction-cost or execution interpretation.

    ### 5.2 Dependence and annual sensitivity

    An annual sensitivity selects the earliest eligible validation origin and then each next origin at least twelve monthly bars later. Only four nonoverlapping paths remain, with origin sessions from July 2021 through July 2024 and target sessions from August 2021 through July 2025. On that subset, annual-return GRU endpoint MSE is 64.53% below zero return and LSTM is 69.04% below. Those different subset errors are not estimates of the full 47-origin comparison; nonoverlap does not establish independence. No interval is claimed for four paths.

    On all 47 annual-return origins, the GRU-minus-zero-return squared-loss interval is [-0.0303697, 0.00529228] with 12-month blocks and [-0.0258706, -0.00546571] with 24-month blocks. Their disagreement about including zero illustrates sensitivity to resampling choices. It does not provide an independent significance result. The supplement retains both block choices for every task and every method, including degenerate cases whose interval bounds are undefined.

    ### 5.3 Corporate actions and post-hoc seed analysis

    Twenty-one overlapping monthly H21 paths cross the split effective August 31, 2020. They account for 74.01% of nominal zero-return baseline squared error and 76.91% of price persistence squared error. They represent one event. Excluding these paths changes the monthly nominal-return GRU improvement from 0.08% overall to 3.60% on the remaining 1,949 origins. That retrospective group comparison does not establish a selection rule for future events.

    A label-only split adjustment reduces those windows' share of zero-return squared error to 1.58%. The prediction is unchanged; the outcome definition changes. Subsequently, all eight methods were retrained on the same five monthly folds using split-neutral cumulative price returns. Historical log-relative windows matched the original histories, while anchors and targets changed. The first-seed GRU improvement was 3.96%, and paired 21- and 42-session descriptive intervals included zero.

    Two additional seeds were fixed after that result was inspected. GRU improvements were 3.96%, -24.67% and 3.99% for seeds 2026, 2027 and 2028. Table 2 separates the mean loss across seeds from the loss of predictions averaged across seeds, both on the same 1,970 origins. Averaging predictions can reduce squared loss even when individual runs are unstable. The LSTM prediction mean has 5.70% lower MSE than zero return, while its mean seed loss is worse. No ensemble confidence interval or independent confirmation was added; these are post-hoc calculations on existing predictions.

    **Table 2.** Post-hoc split-neutral monthly relative endpoint-MSE improvements. Both columns compare with zero return; “Prediction mean” scores the arithmetic mean of the three predicted paths.

    {seed_table}

    ### 5.4 Post-hoc price anchoring

    A separate diagnostic added a zero-initialized residual head to each neural price predictor, allowing epoch-zero persistence in inner stopping. It reused the same 25 historical price folds: quarterly H3, monthly H21 and hourly H1/H4/H8. This changes capacity, initialization and stopping options, so it does not isolate a single causal architectural effect. Inner stopping selected persistence in 40 of 100 model-fold fits. Quarterly pooled improvements ranged from 2.00% to 9.05% across the four neural methods, but used only 52 overlapping origins. Hourly errors remained above persistence; monthly GRU/LSTM improvements were 0.24%/0.20% on the split-sensitive nominal target. These diagnostics remain separate from the unchanged core matrix.

    ## 6. Discussion

    The comparison shows how a persistence reference, output recipe and target convention change the interpretation of financial forecast scores. Under the recorded recipes, direct nominal-price neural predictions often have much larger endpoint losses than persistence. Return prediction is closer to its zero reference for many intraday tasks, while apparent long-horizon improvements rely on fewer overlapping outcomes. These patterns do not rank all possible implementations of the architectures. They describe finite-budget recipes with fixed early stopping and a single core seed.

    Corporate-action sensitivity is substantial at the monthly scale because raw nominal paths contain a split discontinuity. A method that approximates a split-affected level can appear strong on those paths and poor elsewhere. Split-neutral targets address a different statistical quantity; their improvement cannot be attributed to forecasting a future split. They still exclude dividends. Likewise, exponentiating a predicted log return gives a price-scale transformation, not automatically a conditional mean price under squared loss. A comparison across target representations requires common evaluation units specified in advance.

    Overlap, unequal histories and sample dependence remain central. Thousands of minute origins and 47 annual origins do not represent comparable amounts of independent information. A three-month protected interval cannot supply a complete twelve-month label. Historical vendor revisions may also differ from values actually available at the decision time, even when label boundaries are correct. Source identity and calendar completeness establish what was analyzed; they do not establish a fully point-in-time trading simulation.

    All reported periods were already exposed during development. During source investigation, a provider page also displayed a current AAPL quote with an unverified effective timestamp; it was not incorporated into features or scores. The investigator therefore cannot claim complete blindness to subsequent market information. The 2026Q3 series remains sealed and unscored. No trading strategy, transaction costs or profitability is evaluated, and lower forecast MSE alone would not establish a tradable advantage.

    Reproducibility is assessed separately from predictive value. {restoration_text} These checks assess the saved artifacts and interface. They do not make the historical comparisons independent or establish a forecasting gain. Compute receipts retain pauses and parallel stage envelopes; these are not complete measurements of active per-model CPU time. No GPU training or new tuning is used in the approved continuation.

    ## 7. Conclusion

    This study compares price and cumulative-return path forecasting for AAPL under matched historical-origin rules and explicit persistence baselines. Among {nprice} neural comparisons on the {len(price)} complete price tasks, {price_wins} have lower pooled endpoint MSE than persistence. Long-horizon return improvements are limited by few overlapping outcomes, and monthly comparisons are materially affected by one split event and post-hoc seed instability. The evidence supports careful target definitions and baseline reporting; it does not establish independent or profitable forecasting performance. Independent confirmation requires a future frozen protocol and sufficient eligible outcomes.

    ## Data and code availability

    The accompanying supplementary report retains all twenty task tables, fold and step errors, descriptive intervals, realized training histories, source hashes and restoration audits. Its source and package receipts identify the code commit and local reproducibility artifacts. Raw inputs require the matching source versions and applicable permission; the researcher-confirmed scope is recorded separately.{' The complete core has 100 folds, 400 neural fits and a twenty-task inference package.' if complete else ' This layout preview is not a complete result release; its older inference-package snapshot remains partial.'} No archival DOI has been assigned, and nothing has been submitted externally.

    ## References

    {refs}
    """)
    # dedent cannot remove indentation from injected Markdown tables; normalize
    # only the template's four spaces and keep tables/references unindented.
    paper = "\n".join(line[4:] if line.startswith("    ") else line for line in paper.splitlines()).strip() + "\n"
    check(len(cells) == 140 and len(baseline_fields) == 10, "Main numerical inventory differs")
    body, bibliography = paper.split("## References\n\n", 1)
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", body)}
    listed = {int(n) for n in re.findall(r"^\[(\d+)\]", bibliography, flags=re.M)}
    check(cited == listed == set(range(1, 13)), "References must actually be cited")
    output = args.output.resolve()
    check(not output.exists(), "Preserve previous compact-paper releases")
    output.mkdir(parents=True)
    (output / "MAIN_PAPER_EN.md").write_text(paper, encoding="utf8")
    shutil.copy2(args.authors, output / "AUTHOR_INFORMATION.json")
    shutil.copytree(BASE / "author_affiliation_sources/20261108", output / "author_affiliation_sources")
    (output / "RESULTS_SOURCE.json").write_text(json.dumps({"path": str(source), "sha256": sha(source)}, indent=2)+"\n")
    figures = output / "figures"
    figures.mkdir()
    with (figures / "short_paper_error_ratios.csv").open("x", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    facts = {"status": "compact_paper_source_prepared", "at_utc": datetime.now(timezone.utc).isoformat(),
             "layout_preview": args.layout_preview, "complete_core": complete,
             "completed_folds": results["completed_folds"], "complete_tasks": results["complete_tasks"],
             "authors_ready": authors_ready, "author_information_sha256": sha(args.authors),
             "author_name_confirmed": authors_ready,
             "email_self_fill": any(a.get("email_completion") == "author_will_fill" for a in authors["authors"]),
             "source_results": str(source), "source_results_sha256": sha(source),
             "supplement": str(args.supplement.resolve()) if args.supplement else None,
             "price_neural_comparisons": nprice, "price_neural_lower_mse": price_wins,
             "return_neural_comparisons": nreturn, "return_neural_lower_mse": return_wins,
             "baseline_rows": baseline_fields, "chart_cells": 140, "references": 12,
             "checked_parent_reference_ids": list(reference_ids),
             "seed_aggregation_sha256": sha(seed_dir / "SEED_VERSUS_ENSEMBLE.csv"),
             "seed_aggregation_crosscheck_sha256": sha(seed_dir / "SEED_AGGREGATION_CROSSCHECK.json"),
             "core_seed": 2026, "posthoc_seeds": [2026, 2027, 2028],
             "initial_plan_principal_horizons": [["annual", 12], ["quarterly", 3], ["monthly", 21], ["hourly", 8], ["minute", 60]],
             "initial_plan_principal_task_count": 10, "auxiliary_task_count": 10,
             "main_words_including_references": len(re.findall(r"\b[\w’'-]+\b", paper)),
             "new_training_started": False, "independent_test": False, "Q3_values_read": False,
             "externally_published": False, "goal_marked_complete": False,
             "artifacts": {"MAIN_PAPER_EN.md": sha(output / "MAIN_PAPER_EN.md"),
                           "figures/short_paper_error_ratios.csv": sha(figures / "short_paper_error_ratios.csv")},
             "script_sha256": sha(__file__)}
    (output / "MAIN_PAPER_SOURCE_AUDIT.json").write_text(json.dumps(facts, indent=2)+"\n")
    print(json.dumps({k: facts[k] for k in ("status", "completed_folds", "authors_ready", "main_words_including_references")}))


if __name__ == "__main__":
    main()
