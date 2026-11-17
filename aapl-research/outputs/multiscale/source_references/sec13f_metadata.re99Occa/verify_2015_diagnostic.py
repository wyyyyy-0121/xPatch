"""Independently recheck a fixed 2015 third-party 13F diagnostic."""
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import statistics


here = Path(__file__).resolve().parent
out = here.parents[1]
receipt_file = here / "2015_13F_DIAGNOSTIC_V2.json"
receipt = json.loads(receipt_file.read_text(encoding="utf8"))
if receipt.get("status") != "secondary_mirror_diagnostic_not_admitted" or receipt.get("data_admitted") is not False:
    raise ValueError("This verifier only accepts the unadmitted 2015 diagnostic")
if hashlib.sha256((here / "audit_2015_13f.py").read_bytes()).hexdigest() != receipt["audit_source_sha256"]:
    raise ValueError("The recorded diagnostic source changed")


def close_at(path, date_key, close_key):
    with path.open(encoding="utf8", newline="") as stream:
        selected = []
        for row in csv.DictReader(stream):
            raw_date = row[date_key]
            try:
                day = datetime.strptime(raw_date, "%Y-%m-%d").date().isoformat()
            except ValueError:
                day = datetime.strptime(raw_date, "%d-%m-%Y").date().isoformat()
            if day == "2015-09-30":
                selected.append(Decimal(row[close_key]))
    if len(selected) != 1:
        raise ValueError("Expected exactly one historical candidate row")
    return selected[0]


kaggle = out / "kaggle_candidates"
alpha = kaggle / "pratham_20261107T062051911459Z"
adjusted = kaggle / "kpatel_20261107T054630755811Z"
plotly = out / "source_references/plotly_20261107T063711607888Z"
for directory, expected_hash in ((alpha, receipt["candidate_diagnostic"]["alpha_receipt_sha256"]),
                                 (adjusted, receipt["candidate_diagnostic"]["kpatel_receipt_sha256"]),
                                 (plotly, receipt["candidate_diagnostic"]["plotly_receipt_sha256"])):
    if hashlib.sha256((directory / "receipt.json").read_bytes()).hexdigest() != expected_hash:
        raise ValueError("A candidate receipt changed")
for directory in (alpha, adjusted):
    source = json.loads((directory / "receipt.json").read_text(encoding="utf8"))
    for item in source["files"]:
        if hashlib.sha256((directory / item["file"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("A candidate file changed")
reference_receipt = json.loads((plotly / "receipt.json").read_text(encoding="utf8"))
if hashlib.sha256((plotly / "reference.csv").read_bytes()).hexdigest() != reference_receipt["reference_sha256"]:
    raise ValueError("The pinned reference file changed")
factor = Decimal(1)
with (adjusted / "candidate_02.csv").open(encoding="utf8", newline="") as stream:
    for row in csv.DictReader(stream):
        if "2015-09-30" < row["Date"] <= "2026-06-30":
            left, right = row["Stock Splits"].split(":")
            factor *= Decimal(left) / Decimal(right)
alpha_value = close_at(alpha / "candidate_01.csv", "Date", "Close Price")
adjusted_value = close_at(adjusted / "candidate_01.csv", "Date", "Close") * factor
reference_value = close_at(plotly / "reference.csv", "Date", "AAPL.Close")
for key, value in (("alpha_attributed_nominal_close", alpha_value),
                   ("kpatel_split_adjusted_times_reported_factor", adjusted_value),
                   ("plotly_pinned_2017_nominal_close", reference_value)):
    if not math.isclose(receipt["candidate_diagnostic"][key], float(value), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Candidate conversion differs: {key}")
table = json.loads((here / "aapl_2015q3_table.json").read_text(encoding="utf8"))["data"]
large = [(row[0][1], Decimal(1000) * Decimal(row[2]) / Decimal(row[3])) for row in table
         if row[1][0] == "2015-09-30" and not row[4] and type(row[2]) is int and
         type(row[3]) is int and row[2] > 0 and row[3] >= 10_000_000]
summary = receipt["posthoc_large_holding_summary"]
if (len(large) != summary["rows"] or len({x[0] for x in large}) != summary["distinct_manager_links"] or
        not math.isclose(float(statistics.median(x[1] for x in large)), summary["median_implied_usd"], abs_tol=1e-10) or
        sum(abs(price - reference_value) <= Decimal("0.01") for _, price in large) != summary["rows_within_0_01_usd_of_plotly"] or
        sum(abs(price - alpha_value) <= Decimal("0.01") for _, price in large) != summary["rows_within_0_01_usd_of_alpha"]):
    raise ValueError("Cross-sectional mirror summary differs")
for name, row in receipt["selected_filings"].items():
    detail_path = here / f"{name}_detailed_2015q3.json"
    if hashlib.sha256(detail_path.read_bytes()).hexdigest() != row["mirror_detail_sha256"]:
        raise ValueError("A mirror detail file changed")
    data = json.loads(detail_path.read_text(encoding="utf8"))["data"]
    matching = [r for r in data if len(r) == 14 and r[0] == "AAPL" and r[1] == "APPLE INC" and
                r[3] == "037833100" and r[7] is None and not r[8] and
                type(r[4]) is int and type(r[6]) is int and r[4] > 0 and r[6] > 0]
    value, shares = sum(r[4] for r in matching), sum(r[6] for r in matching)
    if (len(matching) != row["mirror_detail_rows"] or value != row["value_thousands_usd"] or
            shares != row["reported_shares"] or
            not math.isclose(float(Decimal(1000) * value / shares), row["implied_usd_per_share"], abs_tol=1e-10)):
        raise ValueError(f"The mirror detail calculation differs: {name}")
result = {"status": "internal_diagnostic_rechecked", "created_at_utc": datetime.now(timezone.utc).isoformat(),
          "data_admitted": False, "formal_experiment": False, "Q3_2026_prices_requested": False,
          "filings_checked": len(receipt["selected_filings"]), "large_holding_rows_checked": len(large),
          "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "diagnostic_sha256": hashlib.sha256(receipt_file.read_bytes()).hexdigest(),
          "limitations": ["The calculation reuses one third-party mirror; it does not verify the original SEC filing.",
                          "Candidate source provenance and publication rights remain unresolved."]}
with (here / "INDEPENDENT_CHECK.json").open("x", encoding="utf8") as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
print(json.dumps({"status": result["status"], "filings_checked": result["filings_checked"],
                  "large_holding_rows_checked": result["large_holding_rows_checked"]}))
