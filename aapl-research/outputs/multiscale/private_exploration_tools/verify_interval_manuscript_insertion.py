"""Check a private manuscript's interval insertion against saved source results."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import csv
import json
import math
from pathlib import Path
import sys


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


root = Path(sys.argv[1]).resolve()
audit_path = root / "PRIVATE_ANNUAL_MANUSCRIPT_AUDIT.json"
audit = json.loads(audit_path.read_text(encoding="utf8"))
md = root / "PRIVATE_WORKING_MANUSCRIPT_EN.md"
parent = Path(audit["parent_directory"]) / md.name
require(digest(md) == audit["manuscript_sha256"] and
        digest(parent) == audit["parent_manuscript_sha256"], "Manuscript identity differs")

text = md.read_text(encoding="utf8")
heading = "### Descriptive uncertainty across completed tasks\n"
marker = "### Split sensitivity and label definition"
require(text.count(heading) == 1 and text.count(marker) == 1, "Section boundary differs")
start = text.index(heading)
end = text.index(marker, start)
section = text[start:end]
new_sentence = ("For the selected windows, the GRU endpoint MSE is 64.53% lower than "
                "the zero-return baseline, but the small set cannot establish a stable advantage.")
old_sentence = ("The selected GRU return result is 64.53% relative to zero return, but the small "
                "set cannot establish a stable advantage.")
reconstructed = text[:start] + text[end:]
require(reconstructed.count(new_sentence) == 1 and
        reconstructed.replace(new_sentence, old_sentence, 1) == parent.read_text(encoding="utf8"),
        "Unrelated parent manuscript text changed")

interval_path = root / "PRIVATE_INTERVALS_ALL_TASKS.csv"
pooled_path = root / "PRIVATE_METRICS_POOLED.csv"
annual_path = root / "PRIVATE_ANNUAL_NONOVERLAP_RESULTS.csv"
for path in (interval_path, pooled_path, annual_path):
    require(digest(path) == audit["csv_sha256"][path.name], "A copied results file changed")
with interval_path.open(encoding="utf8", newline="") as f:
    intervals = list(csv.DictReader(f))
with pooled_path.open(encoding="utf8", newline="") as f:
    pooled = list(csv.DictReader(f))
with annual_path.open(encoding="utf8", newline="") as f:
    annual = list(csv.DictReader(f))

metrics = {(r["profile"], r["horizon"], r["target"], r["model"]): r for r in pooled}
models = {"last_close", "drift", "seasonal", "ridge", "mlp", "gru", "lstm", "xpatch"}
blocks = {"annual": {12, 24}, "quarterly": {3, 6}, "monthly": {21, 42}, "hourly": {2, 5}}
tasks = {(r["profile"], r["horizon"], r["target"]) for r in intervals}
require(len(intervals) == 192 and len(tasks) == 12 and len(metrics) == 96,
        "The task, model, or interval row count differs")
keys = set()
for row in intervals:
    task = (row["profile"], row["horizon"], row["target"])
    key = (*task, row["model"], int(row["block_length"]))
    require(key not in keys and row["model"] in models and
            int(row["block_length"]) in blocks[row["profile"]], "Unexpected or duplicate interval row")
    keys.add(key)
    actual = float(metrics[(*task, row["model"])]["endpoint_mse"])
    baseline = float(metrics[(*task, "last_close")]["endpoint_mse"])
    require(math.isclose(float(row["mean_model_minus_baseline_loss"]), actual - baseline,
                         rel_tol=1e-10, abs_tol=1e-15), "Paired loss does not match pooled results")
    require(row["resamples"] == "5000" and row["seed"] == "2026", "Resampling setup differs")
    if row["status"] == "descriptive_development_only":
        require(math.isfinite(float(row["lower"])) and math.isfinite(float(row["upper"])) and
                float(row["lower"]) < float(row["upper"]), "Invalid interval")
    else:
        require(row["status"] == "degenerate_bootstrap" and not row["lower"] and not row["upper"],
                "Degenerate interval must be blank")
counts = Counter(row["status"] for row in intervals)
require(counts == {"descriptive_development_only": 164, "degenerate_bootstrap": 28},
        "Interval status totals differ")

example = sorted((r for r in intervals if r["profile"] == "annual" and r["target"] == "future_log_return"
                  and r["model"] == "gru"), key=lambda r: int(r["block_length"]))
require(len(example) == 2 and [r["block_length"] for r in example] == ["12", "24"] and
        all(r["origins"] == "47" for r in example), "Annual example identity differs")
for row in example:
    rendered = f"[{float(row['lower']):.6g}, {float(row['upper']):.6g}]"
    require(rendered in section, "Annual example numbers differ from the CSV")
selected = [r for r in annual if r["target"] == "future_log_return" and r["model"] == "gru"]
require(len(selected) == 1 and f"{100 * float(selected[0]['relative_improvement']):.2f}%" == "64.53%",
        "The separate four-window annual improvement differs")
for phrase in ("12 completed five-fold tasks", "192 task–model–block rows", "no development p-values",
               "47 overlapping origins", "not an independent significance claim", "four-window nonoverlap"):
    require(phrase in section, "A sample-size or limitation statement is missing")

out = root / "PRIVATE_INTERVAL_INSERTION_VERIFICATION.json"
require(not out.exists(), "Verification receipt already exists")
receipt = {
    "status": "private_interval_manuscript_insertion_independently_checked",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "manuscript_sha256": digest(md), "parent_sha256": digest(parent),
    "interval_csv_sha256": digest(interval_path), "pooled_csv_sha256": digest(pooled_path),
    "annual_csv_sha256": digest(annual_path), "script_sha256": digest(Path(__file__)),
    "checks": {"tasks": len(tasks), "rows": len(intervals), "paired_means_match_pooled_metrics": True,
               "degenerate_rows_blank": counts["degenerate_bootstrap"],
               "annual_intervals_and_separate_nonoverlap_result_match": True,
               "other_parent_text_preserved": True, "post_exposure_limitations_present": True},
    "limitations": ["Descriptive intervals do not establish independent significance or publication rights."]
}
out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
print(json.dumps({"status": receipt["status"], "path": str(out), "checks": receipt["checks"]}, ensure_ascii=False))
