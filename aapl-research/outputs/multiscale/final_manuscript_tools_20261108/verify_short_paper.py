"""Cross-check a compact paper against its source results and rendered PDF."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import subprocess
import sys

from docx import Document
from docx.oxml.ns import qn
from format_short_paper import inspect
from build_short_paper import TASKS, TARGETS, MODELS, MODEL_NAMES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    folder = Path(sys.argv[1]).resolve()
    audit = json.loads((folder / "MAIN_PAPER_SOURCE_AUDIT.json").read_text())
    source = Path(audit["source_results"])
    assert sha(source) == audit["source_results_sha256"]
    result = json.loads(source.read_text())
    tasks = {(t["profile"], t["horizon"], t["target"]): t for t in result["tasks"]}
    markdown = (folder / "MAIN_PAPER_EN.md").read_text()
    assert sha(folder / "MAIN_PAPER_EN.md") == audit["artifacts"]["MAIN_PAPER_EN.md"]
    raw, word, pdf = [folder / name for name in ("MAIN_PAPER_RAW.docx", "MAIN_PAPER_EN.docx", "MAIN_PAPER_EN.pdf")]
    content = inspect(word)
    assert content == inspect(raw)
    doc = Document(word)
    assert len(doc.tables) == 2 and len(doc.inline_shapes) == 1
    columns = [int(s._sectPr.find(qn("w:cols")).get(qn("w:num"), "1")) for s in doc.sections]
    assert columns == [1, 2, 1, 2, 2]
    text = " ".join(content["text"])
    assert "Principal horizons in the initial plan" in text
    assert "annual H12, quarterly H3, monthly H21, hourly H8 and minute H60" in text
    assert "the remaining horizons and full-path metrics are auxiliary" in text
    assert audit["initial_plan_principal_task_count"] == 10 and audit["auxiliary_task_count"] == 10
    author = json.loads((folder / "AUTHOR_INFORMATION.json").read_text())
    for a in author["authors"]:
        assert a["name_en"] in text
        if a.get("email"):
            assert a["email"] in text
        else:
            assert a["email_completion"] == "author_will_fill" and "[Email to be added by the author]" in text
    for term in ("High School Affiliated to Shanghai Jiao Tong University", "Shanghai, 200439, China"):
        assert term in text
    # Recompute each baseline row directly from the result file, independently
    # of the writer's display inventory.
    rows = doc.tables[0].rows
    assert len(rows) == 11
    baseline_checks = []
    for i, (profile, horizon) in enumerate(TASKS, 1):
        p, r = [tasks[profile, horizon, target] for target in TARGETS]
        np_, nr = p["pooled"]["origins"], r["pooled"]["origins"]
        origins = f"{np_:,}" if np_ == nr else f"{np_:,}/{nr:,}"
        folds = "" if audit["complete_core"] else f" ({len(p['completed_folds'])}/{len(r['completed_folds'])})"
        expected = [profile.title()+f" H{horizon}"+folds, origins,
                    f"{p['pooled']['models']['last_close']['endpoint_mse']:.4g}",
                    f"{r['pooled']['models']['last_close']['endpoint_mse']:.4g}"]
        assert [c.text for c in rows[i].cells] == expected
        baseline_checks.append(expected)
    figure_csv = folder / "figures/short_paper_error_ratios.csv"
    assert sha(figure_csv) == audit["artifacts"]["figures/short_paper_error_ratios.csv"]
    with figure_csv.open(newline="") as stream:
        cells = list(csv.DictReader(stream))
    keys = set()
    for cell in cells:
        key = (cell["profile"], int(cell["horizon"]), cell["target"], cell["model"])
        assert key not in keys
        keys.add(key)
        task = tasks[key[:3]]
        mse = task["pooled"]["models"]
        assert math.isclose(float(cell["ratio"]), mse[key[3]]["endpoint_mse"]/mse["last_close"]["endpoint_mse"], rel_tol=1e-13)
        assert int(cell["completed_folds"]) == len(task["completed_folds"])
    assert keys == {(p, h, t, m) for p, h in TASKS for t in TARGETS for m in MODELS[1:]}
    figure_audit = json.loads((folder / "figures/FIGURE_AUDIT.json").read_text())
    assert figure_audit["source_sha256"] == sha(figure_csv)
    image_sha = sha(folder / "figures/short_paper_error_ratios.png")
    assert content["media"] == [image_sha] and figure_audit["artifacts"]["short_paper_error_ratios.png"] == image_sha
    seed_csv = Path(__file__).resolve().parents[1] / "posthoc_seed_aggregation/20261108_complete_reporting/SEED_VERSUS_ENSEMBLE.csv"
    assert sha(seed_csv) == audit["seed_aggregation_sha256"]
    with seed_csv.open(newline="") as stream:
        seeds = list(csv.DictReader(stream))
    assert len(doc.tables[1].rows) == 9
    for i, row in enumerate(seeds, 1):
        expected = [MODEL_NAMES[i-1], f"{100*float(row['mean_seed_relative_improvement']):.2f}%",
                    f"{100*float(row['ensemble_relative_improvement']):.2f}%"]
        assert [c.text for c in doc.tables[1].rows[i].cells] == expected
    core_price = [t for t in tasks.values() if t["target"] == "future_close" and not t["missing_folds"]]
    core_return = [t for t in tasks.values() if t["target"] == "future_log_return" and not t["missing_folds"]]
    for kind, ts in (("price", core_price), ("return", core_return)):
        wins = sum(t["pooled"]["models"][m]["endpoint_mse"] < t["pooled"]["models"]["last_close"]["endpoint_mse"]
                   for t in ts for m in MODELS[4:])
        assert audit[kind+"_neural_comparisons"] == 4*len(ts) and audit[kind+"_neural_lower_mse"] == wins
    body, refs = markdown.split("## References\n\n", 1)
    assert {int(n) for n in re.findall(r"\[(\d+)\]", body)} == set(range(1, 13))
    assert {int(n) for n in re.findall(r"^\[(\d+)\]", refs, re.M)} == set(range(1, 13))
    assert len(content["links"]) == 13 and len(content["math"]) == 20
    if audit["layout_preview"]:
        assert not audit["complete_core"] and "experiments incomplete" in text
        assert result["completed_folds"] < 100
    else:
        assert audit["complete_core"] and result["completed_folds"] == 100 and result["complete_tasks"] == 20
        assert "experiments incomplete" not in text and "older inference-package snapshot" not in text
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M)[1])
    assert 4 <= pages <= 6, f"User requested 4–6 pages; rendered {pages}"
    assert "A4" in info and re.search(r"Page rot:\s+0", info)
    extracted = folder / "MAIN_PAPER_EN_PDF_TEXT.txt"
    subprocess.run(["pdftotext", "-layout", str(pdf), str(extracted)], check=True)
    pdf_text = extracted.read_text()
    collapsed = re.sub(r"\s+", "", pdf_text).replace("−", "-")
    for a in author["authors"]:
        assert re.sub(r"\s+", "", a["name_en"]) in collapsed
    assert "200439,China" in collapsed
    for row in baseline_checks:
        assert all(re.sub(r"\s+", "", v) in collapsed for v in row[1:])
    assert "[Emailtobeaddedbytheauthor]" in collapsed if audit["email_self_fill"] else True
    footer = "LAYOUT PREVIEW" if audit["layout_preview"] else "HISTORICAL DEVELOPMENT"
    pdf_pages = [p for p in pdf_text.split("\f") if p.strip()]
    assert len(pdf_pages) == pages and all(footer in p for p in pdf_pages)
    fonts = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True, check=True).stdout
    assert "TimesNewRomanPSMT" in fonts and "TimesNewRomanPS-BoldMT" in fonts
    (folder / "PDF_FONT_INVENTORY.txt").write_text(fonts)
    receipt = {"status": "compact_paper_word_and_pdf_content_checked", "at_utc": datetime.now(timezone.utc).isoformat(),
               "pdf_pages": pages, "layout_preview": audit["layout_preview"], "complete_core": audit["complete_core"],
               "completed_folds": audit["completed_folds"], "columns": columns, "baseline_rows_checked": 10,
               "initial_plan_principal_task_count": 10, "auxiliary_task_count": 10,
               "chart_ratio_cells_checked": 140, "seed_versus_ensemble_rows_checked": 8,
               "editable_math_objects": len(content["math"]), "cited_references": 12, "reference_hyperlinks": 13,
               "author_name_confirmed": audit["author_name_confirmed"], "email_self_fill": audit["email_self_fill"],
               "body_content_preserved": True, "scientific_figure_bytes_verified": True,
               "rendered_portrait_A4": True, "body_font_embedded_in_pdf": "Times New Roman",
               "visual_review_complete": False, "new_training_started": False, "Q3_values_read": False,
               "externally_submitted": False, "goal_marked_complete": False,
               "artifacts": {p.name: sha(p) for p in (raw, word, pdf, extracted)},
               "source_audit_sha256": sha(folder / "MAIN_PAPER_SOURCE_AUDIT.json"), "script_sha256": sha(__file__)}
    (folder / "MAIN_PAPER_CONTENT_VERIFICATION.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"status": receipt["status"], "pdf_pages": pages, "ratios_checked": 140,
                      "author": [a["name_en"] for a in author["authors"]], "email_self_fill": audit["email_self_fill"]}))


if __name__ == "__main__":
    main()
