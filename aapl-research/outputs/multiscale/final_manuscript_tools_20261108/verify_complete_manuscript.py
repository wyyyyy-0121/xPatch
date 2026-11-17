"""Check complete manuscript numbers, document conversion and rendered text."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
from xml.etree import ElementTree as ET
from zipfile import ZipFile

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
MODELS = ("last_close", "drift", "seasonal", "ridge", "mlp", "gru", "lstm", "xpatch")
NAMES = ("Persistence / zero", "Drift", "Seasonal", "Ridge", "MLP", "GRU", "LSTM", "xPatch")
ORDER = ("annual", "quarterly", "monthly", "hourly", "minute")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def command(arguments):
    return subprocess.run(arguments, capture_output=True, text=True, check=True, timeout=120).stdout


def tables(text):
    result, current = [], []
    for line in text.splitlines():
        if line.startswith("|"):
            if re.match(r"^\|\s*:?-+", line):
                continue
            current.append([s.strip().replace("\\(", "").replace("\\)", "") for s in line.strip("|").split("|")])
        elif current:
            result.append(current)
            current = []
    if current:
        result.append(current)
    return result


def section(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]


def document(path):
    with ZipFile(path) as archive:
        body = ET.fromstring(archive.read("word/document.xml"))
        relations = ET.fromstring(archive.read("word/_rels/document.xml.rels"))
        content = []
        for table in body.findall(".//w:tbl", NS):
            content.append([["".join(x.text or "" for x in cell.iter() if x.tag in (
                f"{{{NS['w']}}}t", f"{{{NS['m']}}}t")).strip() for cell in row.findall("./w:tc", NS)]
                            for row in table.findall("./w:tr", NS)])
        return {"text": [x.text or "" for x in body.findall(".//w:t", NS)],
                "math": ["".join(x.itertext()) for x in body.findall(".//m:oMath", NS)],
                "tables": content, "drawings": len(body.findall(".//w:drawing", NS)),
                "links": sorted(x.attrib.get("Target", "") for x in relations if x.attrib.get("Type", "").endswith("/hyperlink")),
                "media": sorted(hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist() if n.startswith("word/media/"))}


def main():
    root = Path(sys.argv[1]).resolve()
    build = load(root / "MANUSCRIPT_BUILD_AUDIT.json")
    parent = Path(build["parent"])
    evidence = Path(build["evidence"])
    md, raw, word, pdf = (root / n for n in ("MANUSCRIPT_EN.md", "MANUSCRIPT_RAW.docx", "MANUSCRIPT_EN.docx", "MANUSCRIPT_EN.pdf"))
    require(digest(md) == build["markdown_sha256"], "Manuscript changed after evidence integration")
    require(digest(evidence / "ALL100_EVIDENCE_RECEIPT.json") == build["evidence_receipt_sha256"], "Core evidence changed")
    receipt = load(evidence / "ALL100_EVIDENCE_RECEIPT.json")
    for name, expected in receipt["artifacts"].items():
        require(digest(evidence / name) == expected, "Core artifact changed: " + name)
    require(build["completed_folds"] == 100 and build["complete_tasks"] == 20, "Incomplete manuscript source")
    body = md.read_text(encoding="utf8")
    before, after = document(raw), document(word)
    for field in ("text", "math", "tables", "links", "media"):
        require(before[field] == after[field], "Word formatting changed " + field)
    require((len(after["math"]), len(after["links"]), len(after["tables"]), after["drawings"]) == (30, 18, 37, 2), "Unexpected document structure")
    require(tables(body) == after["tables"], "Markdown and Word table cells differ")
    results = load(root / "all_task_results/RESULTS.json")
    require(digest(root / "all_task_results/RESULTS.json") == receipt["artifacts"]["results/RESULTS.json"], "Numerical source differs")
    for name, expected in results["csv_sha256"].items():
        require(digest(root / "all_task_results" / name) == expected, "Numerical CSV changed")
    tasks = sorted(results["tasks"], key=lambda t: (ORDER.index(t["profile"]), t["horizon"], t["target"]))
    appendix = section(body, "## Appendix A. Detailed historical development results", "## Appendix B. Forecast timing and original fold design")
    outcome_tables = tables(appendix)
    require(len(outcome_tables) == len(re.findall(r"^### ", appendix, re.M)) == 20, "Detailed task tables differ")
    fields = 0
    for task, table in zip(tasks, outcome_tables, strict=True):
        require(len(table) == 9, "Task lacks eight methods")
        require(task["completed_folds"] == [1, 2, 3, 4, 5] and not task["missing_folds"], "Task lacks five folds")
        for model, name, row in zip(MODELS, NAMES, table[1:], strict=True):
            metrics = task["pooled"]["models"][model]
            expected = [name] + [f"{metrics[k]:.6g}" for k in ("endpoint_mse", "endpoint_rmse", "endpoint_mae", "path_mse")]
            expected += ["—" if metrics[k] is None else f"{100 * metrics[k]:.2f}%" for k in (
                "relative_improvement", "direction_coverage", "direction_accuracy_when_active")]
            require(row == expected, "Detailed outcome differs from independent score")
            fields += 7
    summary_fields = 0
    for target, start, end in (("future_log_return", "### Cumulative log-return target", "### Nominal close-price target"),
                               ("future_close", "### Nominal close-price target", "### Realized training budget")):
        summary = tables(section(body, start, end))
        require(len(summary) == 1 and len(summary[0]) == 11, "Summary lacks all ten horizons")
        selected = [t for t in tasks if t["target"] == target]
        for t, row in zip(selected, summary[0][1:], strict=True):
            expected = [f"{t['pooled']['origins']:,}"] + [f"{100*t['pooled']['models'][m]['relative_improvement']:.2f}%" for m in MODELS[1:]]
            require(row[1:] == expected, "Main summary differs")
            summary_fields += 8
    coverage_tables = tables(section(body, "### Development coverage and baseline magnitudes", "### Cumulative log-return target"))
    require(len(coverage_tables) == 2 and all(len(t) == 11 for t in coverage_tables), "Coverage or denominator table shape differs")
    lookup = {(t["profile"], t["horizon"], t["target"]): t for t in tasks}
    for t, row, denominator in zip([t for t in tasks if t["target"] == "future_close"], coverage_tables[0][1:], coverage_tables[1][1:], strict=True):
        require(row[1:] == ["5/5", "5/5"], "Coverage is incomplete")
        r = lookup[t["profile"], t["horizon"], "future_log_return"]
        expected = [f"{t['pooled']['origins']:,}", f"{t['pooled']['models']['last_close']['endpoint_mse']:.6g}", f"{r['pooled']['models']['last_close']['endpoint_mse']:.6g}"]
        require(denominator[1:] == expected and t["pooled"]["origins"] == r["pooled"]["origins"], "Baseline denominator differs")
    with (root / "all_task_results/TRAINING_MODELS.csv").open(newline="", encoding="utf8") as stream:
        training = list(csv.DictReader(stream))
    require(len(training) == 400 and all(r["status"] == "completed" for r in training), "Incomplete training evidence")
    training_tables = tables(section(body, "### Realized training budget", "### Minute-scale results and fold variation"))
    require(len(training_tables) == 1 and len(training_tables[0]) == 7, "Training table differs")
    for scale, row in zip(ORDER + ("total",), training_tables[0][1:], strict=True):
        subset = training if scale == "total" else [r for r in training if r["profile"] == scale]
        expected = [scale.title(), str(len(subset)), str(sum(r["epoch_cap_reached"] == "True" for r in subset)),
                    str(sum(r["stopped_early"] == "True" for r in subset)), f"{statistics.median(int(r['steps']) for r in subset):.6g}"]
        require(row == expected, "Training counts differ from saved histories")
    minute = tables(section(body, "### Minute-scale results and fold variation", "### Annual sensitivity: nonoverlapping forecast windows"))
    h60 = lookup["minute", 60, "future_log_return"]
    require(len(minute) == 1 and len(minute[0]) == 6, "H60 fold table is incomplete")
    for fold, row in zip(h60["folds"], minute[0][1:], strict=True):
        expected = [str(fold["fold"]), f"{fold['origins']:,}", f"{fold['models']['last_close']['endpoint_mse']:.6g}"] + [f"{100*fold['models'][m]['relative_improvement']:.2f}%" for m in MODELS[4:]]
        require(row == expected, "H60 fold numbers differ")
    design = load(root / "forecast_timing/DESIGN_TIMING_AUDIT.json")
    require(design["planned_folds_verified"] == design["completed_saved_fold_metadata_verified"] == 100 and design["paired_price_return_designs_verified"] == 50, "Timing scope differs")
    timing_tables = tables(body.split("## Appendix B. Forecast timing and original fold design", 1)[1])
    timing_tasks = sorted([t for t in design["tasks"] if t["target"] == "future_close"], key=lambda t: (ORDER.index(t["profile"]), t["horizon"]))
    require(len(timing_tables) == 1 and len(timing_tables[0]) == 11, "Timing table shape differs")
    for t, row in zip(timing_tasks, timing_tables[0][1:], strict=True):
        timing = t["completed_origin_timing"]
        expected = [t["profile"].title() + f" H{t['horizon']}", f"{timing['origins']:,}"]
        expected += [f"{timing[k]:.6g}" for k in ("elapsed_hours_min", "elapsed_hours_median", "elapsed_hours_p95", "elapsed_hours_max")]
        expected += [f"{100*timing['cross_session_fraction']:.2f}%"]
        require(row == expected, "Completed timing values differ")
    split = load(root / "monthly_split_groups/SPLIT_GROUP_AUDIT.json")
    split_tables = tables(section(body, "### Split sensitivity and label definition", "A subsequent label-only diagnostic"))
    require(len(split_tables) == 2, "Split-group tables are missing")
    for target, table in zip(("future_close", "future_log_return"), split_tables, strict=True):
        for model, row in zip(MODELS, table[1:], strict=True):
            cross = split["groups"][target]["crosses_split"]["models"][model]
            other = split["groups"][target]["no_split"]["models"][model]
            require(row[1:] == [f"{cross['endpoint_mse']:.6g}", f"{other['endpoint_mse']:.6g}", f"{100*cross['relative_improvement']:.2f}%", f"{100*other['relative_improvement']:.2f}%"], "Split-group number differs")
    old = (parent / "PRIVATE_WORKING_MANUSCRIPT_EN.md").read_text(encoding="utf8")
    for start, end in (("### Annual sensitivity: nonoverlapping forecast windows", "### Descriptive uncertainty across completed tasks"),
                       ("### Post-hoc price anchoring", "## 6. Limitations and conclusion")):
        require(section(body, start, end) == section(old, start, end), "Previously audited sensitivity text changed")
    require(section(body, "### Split sensitivity and label definition", "### Seed averages and ensemble errors") ==
            section(old, "### Split sensitivity and label definition", "### Post-hoc price anchoring"),
            "Original split and individual-seed evidence changed")
    aggregation_path = root / "posthoc_seed_aggregation/SEED_AGGREGATION_AUDIT.json"
    require(digest(aggregation_path) == build["posthoc_seed_aggregation_sha256"], "Seed aggregation source changed")
    aggregation = load(aggregation_path)
    seed_tables = tables(section(body, "### Seed averages and ensemble errors", "### Post-hoc price anchoring"))
    require(len(seed_tables) == 1 and len(seed_tables[0]) == 9, "Seed/ensemble table lacks methods")
    seed_lookup = {r["model"]: r for r in aggregation["rows"]}
    for model, row in zip(MODELS, seed_tables[0][1:], strict=True):
        v = seed_lookup[model]
        require(row[1:] == [f"{v['mean_seed_endpoint_mse']:.6g}", f"{100*v['mean_seed_relative_improvement']:.2f}%",
                            f"{v['ensemble_endpoint_mse']:.6g}", f"{100*v['ensemble_relative_improvement']:.2f}%"],
                "Seed average or ensemble error differs")
    # Independently compare the complete ratio figure CSV to the scoring JSON.
    fig = load(root / "figures/FIGURE_AUDIT.json")
    for name, expected in fig["artifacts"].items():
        require(digest(root / "figures" / name) == expected, "Figure artifact changed")
    with (root / "figures/completed_task_error_ratios.csv").open(newline="") as stream:
        ratios = list(csv.DictReader(stream))
    require(len(ratios) == 140, "Figure lacks comparisons")
    ratio_keys = set()
    for row in ratios:
        t = lookup[row["profile"], int(row["horizon"]), row["target"]]
        metrics = t["pooled"]["models"]
        key = (row["profile"], row["horizon"], row["target"], row["model"])
        require(key not in ratio_keys, "Duplicate figure cell")
        ratio_keys.add(key)
        require(abs(float(row["endpoint_mse_ratio"]) - metrics[row["model"]]["endpoint_mse"] / metrics["last_close"]["endpoint_mse"]) < 1e-12, "Figure cell differs")
    a = command(["pandoc", "--from=markdown+tex_math_single_backslash+autolink_bare_uris", "--to=plain", "--wrap=none", str(md)])
    b = command(["pandoc", "--from=docx", "--to=plain", "--wrap=none", str(word)])
    pattern = r"[A-Za-z]{3,}(?:[’'-][A-Za-z]+)?"
    ca, cb = Counter(re.findall(pattern, a)), Counter(re.findall(pattern, b))
    for token in ("mathrm", "qquad", "quad", "mspace", "leq"):
        ca.pop(token, None)
        cb.pop(token, None)
    require(ca == cb, "Prose differs between Markdown and Word: " + str((ca - cb, cb - ca)))
    info = command(["pdfinfo", str(pdf)])
    count = int(re.search(r"^Pages:\s+(\d+)\s*$", info, re.M)[1])
    rendered = command(["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), "-"])
    pages = [p for p in rendered.split("\f") if p.strip()]
    require(len(pages) == count and count >= 18 and all("HISTORICAL DEVELOPMENT STUDY" in p for p in pages), "PDF pages or footers differ")
    for fragment in ("100 original task-folds", "400 partition hashes", "119 hours", "Appendix B"):
        require(fragment in rendered, "Rendered PDF lacks " + fragment)
    require("PRIVATE WORKING DRAFT" not in rendered and "85/100" not in rendered, "PDF has stale draft status")
    (root / "RENDERED_TEXT.txt").write_text(rendered, encoding="utf8")
    record = {"status": "complete_manuscript_numbers_and_document_content_verified", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "markdown_sha256": digest(md), "raw_docx_sha256": digest(raw), "formatted_docx_sha256": digest(word), "pdf_sha256": digest(pdf),
              "script_sha256": digest(__file__), "completed_folds": 100, "complete_tasks": 20,
              "checks": {"editable_equations": 30, "hyperlinks": 18, "tables": 37, "figures": 2,
                         "appendix_outcome_fields_verified": fields, "main_summary_numeric_fields_verified": summary_fields,
                         "baseline_denominator_numeric_fields_verified": 30, "training_summary_numeric_fields_verified": 24,
                         "minute_fold_numeric_fields_verified": 30, "timing_numeric_fields_verified": 60,
                         "split_group_numeric_fields_verified": 64, "figure_ratios_verified": len(ratios),
                         "seed_average_and_ensemble_numeric_fields_verified": 32,
                         "all_markdown_table_cells_match_word": True, "formatting_preserves_body_content": True,
                         "prose_word_counts_match": True, "pdf_pages": count, "pdf_footers_verified": True},
              "visual_review_pending": True, "independent_test": False, "Q3_values_read": False}
    with (root / "DOCUMENT_CONTENT_VERIFICATION.json").open("x", encoding="utf8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(record["checks"]))


if __name__ == "__main__":
    main()
