"""Check one pre-cutoff price discrepancy using a third-party 13F mirror.

This diagnostic neither admits a price series nor verifies the mirror against
the original SEC submissions. It does not request or evaluate 2026Q3 prices.
"""
from datetime import datetime, timezone
from html.parser import HTMLParser
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import pandas as pd

from multiscale.audit_alpha_candidates import read_candidate
from multiscale.audit_kaggle_candidates import candidate, kpatel_splits


here = Path(__file__).resolve().parent
out = here.parents[1]
date = "2015-09-30"
cusip = "037833100"
files = {
    "vanguard": ("000095012315011453", "vanguard_2015q3.html", "vanguard_detailed_2015q3.json"),
    "state_street": ("000009375115000232", "state_street_2015q3.html", "state_street_detailed_2015q3.json"),
    "fmr": ("000031506615003305", "fmr_2015q3.html", "fmr_detailed_2015q3.json"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=here / "2015_13F_DIAGNOSTIC.json")
args = parser.parse_args()
target = args.output.resolve()
if not target.is_relative_to(here) or target.parent != here or target.suffix != ".json":
    raise ValueError("The diagnostic receipt must be a new JSON in this local evidence directory")
source_hash = digest(Path(__file__))


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []
        self.sec_links = []
        self.tables = []
        self.headers = []
        self.in_detail = False
        self.in_th = False

    def handle_starttag(self, tag, attrs):
        item = dict(attrs)
        if tag in ("script", "style", "noscript", "svg"):
            self.skip += 1
        if tag == "a" and "sec.gov/Archives/edgar/data/" in item.get("href", ""):
            self.sec_links.append(item["href"])
        if tag == "table":
            self.tables.append({"id": item.get("id"), "data_url": item.get("data-url")})
            self.in_detail = item.get("id") == "filingDetailed"
        if self.in_detail and tag == "th":
            self.in_th = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self.skip:
            self.skip -= 1
        if tag == "th":
            self.in_th = False
        if tag == "table":
            self.in_detail = False

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)
            if self.in_th and data.strip():
                self.headers.append(" ".join(data.split()))


def page(path):
    parsed = Page()
    parsed.feed(path.read_text(encoding="utf8", errors="replace"))
    return parsed, " ".join(" ".join(parsed.parts).split())


header_page, _ = page(here / "state_street_detailed_page.html")
expected_headers = ["Sym", "Issuer Name", "Cl", "CUSIP", "Value ($000)", "%", "Shares", "Principal",
                    "Option Type", "Discretion", "Other Manager", "Sole", "Shared", "None"]
if header_page.headers != expected_headers:
    raise ValueError("The mirror's detailed 13F column headings changed")
table = json.loads((here / "aapl_2015q3_table.json").read_text(encoding="utf8"))["data"]
rows = []
for row in table:
    try:
        manager, filing, value, shares, option = row
        if (filing[0] == date and not option and type(value) is int and type(shares) is int and
                value > 0 and shares >= 10_000_000):
            rows.append({"manager": manager[0], "manager_id": manager[1], "filing": filing[1],
                         "implied_nominal_price": 1000 * value / shares})
    except (TypeError, ValueError, IndexError):
        continue
if not rows:
    raise ValueError("No qualifying 2015 common-share mirror records")

filings = {}
for name, (accession, html_name, data_name) in files.items():
    parsed, visible = page(here / html_name)
    if ("Holdings as of 9/30/2015" not in visible or "Value ($000)" not in visible or
            len(parsed.sec_links) != 1 or accession not in parsed.sec_links[0] or
            not any(t["data_url"] == "/data/13f/" + accession for t in parsed.tables)):
        raise ValueError(f"Mirror filing metadata does not match the 2015 period: {name}")
    detailed = json.loads((here / data_name).read_text(encoding="utf8"))["data"]
    apple = [row for row in detailed if len(row) == len(expected_headers) and row[0] == "AAPL" and
             row[1] == "APPLE INC" and row[3] == cusip and not row[8] and row[7] is None and
             type(row[4]) is int and type(row[6]) is int and row[4] > 0 and row[6] > 0]
    if not apple:
        raise ValueError(f"No common-share 13F detail found for {name}")
    value, shares = sum(row[4] for row in apple), sum(row[6] for row in apple)
    aggregated = [row for row in table if row[1][0] == date and row[1][1].startswith(accession) and not row[4]]
    if len(aggregated) != 1 or aggregated[0][2] != value or aggregated[0][3] != shares:
        raise ValueError(f"Mirror detail and aggregate do not match for {name}")
    filings[name] = {"accession": accession, "sec_index_url": parsed.sec_links[0],
                     "mirror_detail_rows": len(apple), "value_thousands_usd": value,
                     "reported_shares": shares, "implied_usd_per_share": 1000 * value / shares,
                     "mirror_html_sha256": digest(here / html_name), "mirror_detail_sha256": digest(here / data_name)}

kaggle = out / "kaggle_candidates"
pratham_dir, pratham_receipt, raw = read_candidate(kaggle / "pratham_20261107T062051911459Z", "pratham")
kpatel_dir, kpatel_receipt, adjusted = candidate(kaggle / "kpatel_20261107T054630755811Z", "kpatel")
actions = kpatel_splits(kpatel_dir, kpatel_receipt)
factor = 1.0
for action in actions:
    if date < action["date"]:
        factor *= action["ratio"]
if factor != 4.0:
    raise ValueError("The assumed post-2015 split conversion differs")
alpha_value = float(raw.loc[raw.date.eq(date), "close"].item())
kpatel_value = float(adjusted.loc[adjusted.date.eq(date), "close"].item()) * factor
plotly_dir = out / "source_references/plotly_20261107T063711607888Z"
plotly_receipt = json.loads((plotly_dir / "receipt.json").read_text(encoding="utf8"))
if (plotly_receipt["status"] != "diagnostic_not_admitted" or
        digest(plotly_dir / "reference.csv") != plotly_receipt["reference_sha256"] or
        plotly_receipt["commit_date_utc"] != "2017-02-22T16:05:36+00:00"):
    raise ValueError("The pinned 2017 reference identity changed")
reference = pd.read_csv(plotly_dir / "reference.csv")
reference_value = float(reference.loc[reference.Date.eq(date), "AAPL.Close"].item())
values = [row["implied_nominal_price"] for row in rows]
selected_prices = [filings[name]["implied_usd_per_share"] for name in files]
receipt = {
    "status": "secondary_mirror_diagnostic_not_admitted", "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "2015-09-30 common-share 13F value/share cross-check; no training or forecast scoring",
    "Q3_2026_prices_requested": False, "experiment_scored": False, "data_admitted": False,
    "security": {"symbol": "AAPL", "cusip": cusip, "report_date": date},
    "audit_source_sha256": source_hash,
    "mirror_root_url": "https://13f.info/cusip/037833100/2015/3",
    "mirror_table_sha256": digest(here / "aapl_2015q3_table.json"),
    "mirror_page_sha256": digest(here / "aapl_2015q3_13finfo.html"),
    "mirror_detail_headers_sha256": digest(here / "state_street_detailed_page.html"),
    "mirror_public_readme_sha256": digest(here / "github_readme.json"),
    "mirror_formatter_blob_sha1": json.loads((here / "data_formatter.json").read_text())["sha"],
    "sec_identified_filing_head_status": 403, "selected_filings": filings,
    "posthoc_large_holding_summary": {"selection": "report date 2015-09-30, no option, at least 10 million shares",
        "rows": len(rows), "distinct_manager_links": len({row["manager_id"] for row in rows}),
        "median_implied_usd": statistics.median(values), "min_implied_usd": min(values), "max_implied_usd": max(values),
        "rows_within_0_01_usd_of_plotly": sum(abs(v - reference_value) <= .01 for v in values),
        "rows_within_0_01_usd_of_alpha": sum(abs(v - alpha_value) <= .01 for v in values)},
    "candidate_diagnostic": {"alpha_attributed_nominal_close": alpha_value,
        "kpatel_split_adjusted_times_reported_factor": kpatel_value, "post_2015_factor": factor,
        "plotly_pinned_2017_nominal_close": reference_value,
        "selected_filing_implied_range_usd": [min(selected_prices), max(selected_prices)],
        "alpha_receipt_sha256": digest(pratham_dir / "receipt.json"),
        "kpatel_receipt_sha256": digest(kpatel_dir / "receipt.json"),
        "plotly_receipt_sha256": digest(plotly_dir / "receipt.json")},
    "limitations": ["All filing observations come through the same third-party mirror; original SEC files were not obtained.",
        "The mirror reports that it may adjust outlying values by a factor of 1000; deployment code was not verified.",
        "The fetched public formatter source and live detailed-row layout differ, so repository code is not proof of deployment.",
        "The three largest filings and share threshold were selected after initial inspection; the comparison is exploratory.",
        "A 13F value/share ratio is a reported valuation, not a certified closing-price record; reporting errors and rounding occur.",
        "The filings may share underlying market prices, and the historical page includes links to later quarters but no later-quarter price values were requested.",
        "The candidate split conversion and historical vendor provenance remain unverified; no source or publication right is admitted."]}
if digest(Path(__file__)) != source_hash:
    raise ValueError("Diagnostic source changed during execution")
with target.open("x", encoding="utf8") as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
print(json.dumps({"status": receipt["status"], "filings_checked": len(filings),
                  "large_holding_rows": len(rows), "receipt": str(target)}))
