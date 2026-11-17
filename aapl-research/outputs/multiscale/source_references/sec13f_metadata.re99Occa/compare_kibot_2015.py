"""Compare one existing, unadmitted Kibot close with a historical mirror."""
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path


here = Path(__file__).resolve().parent
out = here.parents[1]
source = out / "daily_sources/kibot_guest_20261107T072857615166Z"
source_receipt = source / "receipt.json"
source_csv = source / "kibot_daily_raw.csv"
diagnostic = here / "2015_13F_DIAGNOSTIC_V2.json"
source_meta = json.loads(source_receipt.read_text(encoding="utf8"))
comparison = json.loads(diagnostic.read_text(encoding="utf8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if (source_meta.get("status") != "candidate_internal_diagnostic_only" or
        source_meta.get("source_admitted") is not False or source_meta.get("Q3_values_read") is not False or
        source_meta.get("csv_sha256") != digest(source_csv) or
        comparison.get("status") != "secondary_mirror_diagnostic_not_admitted" or
        comparison.get("audit_source_sha256") != digest(here / "audit_2015_13f.py")):
    raise ValueError("Source or diagnostic identity is not the expected unadmitted version")
with source_csv.open(encoding="utf8", newline="") as stream:
    rows = [row for row in csv.DictReader(stream) if row["date"] == "2015-09-30"]
if len(rows) != 1:
    raise ValueError("Expected exactly one candidate row for the diagnostic date")
close = Decimal(rows[0]["close"])
reference = Decimal(str(comparison["candidate_diagnostic"]["plotly_pinned_2017_nominal_close"]))
mirror = Decimal(str(comparison["posthoc_large_holding_summary"]["median_implied_usd"]))
if not close.is_finite() or close <= 0:
    raise ValueError("Invalid candidate value")
result = {"status": "internal_source_diagnostic_not_admitted", "created_at_utc": datetime.now(timezone.utc).isoformat(),
          "date": "2015-09-30", "Q3_2026_prices_requested": False, "experiment_scored": False,
          "source_admitted": False, "publication_rights_verified": False,
          "kibot_close": float(close), "pinned_2017_reference_close": float(reference),
          "third_party_13f_median_valuation_per_share": float(mirror),
          "absolute_difference_to_pinned_reference": float(abs(close - reference)),
          "absolute_difference_to_mirror_median": float(abs(close - mirror)),
          "source_csv_sha256": digest(source_csv), "source_receipt_sha256": digest(source_receipt),
          "mirror_diagnostic_sha256": digest(diagnostic), "script_sha256": digest(Path(__file__)),
          "limitations": ["The source and mirror remain unadmitted; this is not a published market result.",
                          "Daily close definitions, the 13F valuation method, rounding and revisions may differ.",
                          "Neither a third-party mirror nor a single historical date proves overall data quality or rights."]}
with (here / "KIBOT_CLOSE_DIAGNOSTIC.json").open("x", encoding="utf8") as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
print(json.dumps({"status": result["status"], "matched_rows": len(rows), "source_admitted": False}))
