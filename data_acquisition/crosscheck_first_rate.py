"""Compare retained short FirstRate sample with an independent Alpaca SIP run."""
import argparse
from pathlib import Path

import numpy as np

from finance_v2.data import audit_bars, read_bars
from finance_v2.io import digest, write_json


def compare(alpaca_csv: Path, firstrate_csv: Path, output: Path):
    alpaca_raw, first_raw = read_bars(alpaca_csv), read_bars(firstrate_csv)
    alpaca, alpaca_audit = audit_bars(alpaca_raw)
    first, first_audit = audit_bars(first_raw)
    columns = ("open", "high", "low", "close", "volume")
    left = alpaca[list(columns)].add_suffix("_alpaca")
    right = first[list(columns)].add_suffix("_firstrate")
    joint = left.join(right, how="inner")
    if not len(joint):
        raise ValueError("No overlapping XNYS minute bars")
    result = {"status": "price_concordance_check_complete", "formal_eligible": False,
              "alpaca_csv_sha256": digest(alpaca_csv), "firstrate_csv_sha256": digest(firstrate_csv),
              "alpaca_expected_coverage": alpaca_audit["regular_coverage"],
              "firstrate_expected_coverage": first_audit["regular_coverage"],
              "overlap_minutes": int(len(joint)), "firstrate_minutes": int(len(first)),
              "overlap_fraction": len(joint) / len(first), "fields": {},
              "limitations": ["FirstRate's exact feed, trade-condition rules and price-adjustment identity are not independently established.",
                              "Cross-provider agreement over a short recent sample does not validate older history or publication-time availability.",
                              "Volume differences may reflect trade eligibility, aggregation or provider-specific corrections."]}
    for column in columns[:4]:
        difference = np.abs(joint[f"{column}_alpaca"] - joint[f"{column}_firstrate"])
        bps = difference / joint[f"{column}_alpaca"].abs() * 10000
        result["fields"][column] = {"exact_match_fraction": float((difference < 1e-10).mean()),
            "within_one_cent_fraction": float((difference <= 0.010001).mean()),
            "within_one_basis_point_fraction": float((bps <= 1).mean()),
            "median_absolute_difference_usd": float(difference.median()),
            "p99_absolute_difference_usd": float(difference.quantile(.99)),
            "max_absolute_difference_usd": float(difference.max())}
    difference = np.abs(joint.volume_alpaca - joint.volume_firstrate)
    result["fields"]["volume"] = {"exact_match_fraction": float((difference == 0).mean()),
        "median_absolute_difference_shares": float(difference.median()),
        "p99_absolute_difference_shares": float(difference.quantile(.99)),
        "max_absolute_difference_shares": int(difference.max()),
        "pearson_correlation": float(joint.volume_alpaca.corr(joint.volume_firstrate))}
    result["largest_close_differences"] = [
        {"timestamp": stamp.isoformat(), "alpaca_close": float(joint.loc[stamp, "close_alpaca"]),
         "firstrate_close": float(joint.loc[stamp, "close_firstrate"]),
         "difference_usd": float(abs(joint.loc[stamp, "close_alpaca"] -
                                     joint.loc[stamp, "close_firstrate"]))}
        for stamp in np.abs(joint.close_alpaca - joint.close_firstrate).nlargest(10).index]
    result["provider_concordance_gate"] = bool(result["overlap_fraction"] == 1.0 and
        all(result["fields"][key]["within_one_cent_fraction"] == 1.0 for key in columns[:4]))
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpaca-csv", type=Path, required=True)
    parser.add_argument("--firstrate-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(compare(args.alpaca_csv, args.firstrate_csv, args.output))


if __name__ == "__main__":
    main()
