"""Prepare a focused main edition, inheriting the accepted supplement unchanged.

Standard-library setup only; old docx/formatter imports occur only in render-main.
No market reader, scientific computation, supplement build or packaging entry.
"""
from pathlib import Path
import argparse
import difflib
import hashlib
import importlib.util
import json
import os
import shutil
import sys

BASE=Path(__file__).resolve().parent
SUPPLEMENT_FILES=("SUPPLEMENT_EN.md","SUPPLEMENT_EN.docx","SUPPLEMENT_EN.pdf")
THREAD_KEYS=("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_manifest(value):
    (BASE/"PREPARATION_MANIFEST.json").write_text(json.dumps(value,indent=2)+"\n")


def verified_baseline():
    record=json.loads((BASE/"BASE_SOURCE_LOCKS.json").read_text())
    baseline=Path(record["baseline_directory"]).resolve()
    require(BASE!=baseline and not BASE.is_relative_to(baseline) and not baseline.is_relative_to(BASE),
            "Focused edition must be separate from the accepted directory")
    for name in ("MAIN_PAPER_EN.md","MAIN_PAPER_EN.docx","MAIN_PAPER_EN.pdf","MAIN_PAPER_RAW.docx",
                 *SUPPLEMENT_FILES,"AUTHOR_INFORMATION.json","review","figures"):
        require((BASE/name).resolve().is_relative_to(BASE),"Output symlink escapes the new edition: "+name)
    for name,digest in record["files"].items():require(sha(baseline/name)==digest,"Accepted baseline changed: "+name)
    actual=sorted(p.relative_to(baseline).as_posix() for p in (baseline/"figures").rglob("*") if p.is_file())
    require(actual==record["figure_relative_paths"],"Accepted figure file set changed")
    for key in ("reused_builder","reused_formatter"):
        require(sha(record[key]["path"])==record[key]["sha256"],"Accepted build/format source changed")
    accepted=json.loads((baseline/"review/FINAL_DOCUMENT_ACCEPTANCE.json").read_text())
    require(accepted["status"]==record["baseline_acceptance_status"] and accepted["supplement_pages"]==41,
            "Latest accepted baseline qualification differs")
    return baseline,record


def inherited_supplement(record):
    for name in SUPPLEMENT_FILES:
        require(sha(BASE/name)==record["files"][name],"Inherited supplement changed: "+name)
    return {name:record["files"][name] for name in SUPPLEMENT_FILES}


def prepare():
    baseline,record=verified_baseline()
    manifest=json.loads((BASE/"PREPARATION_MANIFEST.json").read_text())
    require(manifest["baseline_locks_sha256"]==sha(BASE/"BASE_SOURCE_LOCKS.json"),"Baseline lock manifest changed")
    require(manifest["status"]=="source_only_builder_not_prepared" and manifest["prepared"] is False,
            "Refuse repeated preparation or overwrite")
    targets=("MAIN_PAPER_EN.md","MAIN_PAPER_EN.docx","MAIN_PAPER_EN.pdf","MAIN_PAPER_RAW.docx",
             *SUPPLEMENT_FILES,"AUTHOR_INFORMATION.json","review/DOCUMENT_BUILD.json","figures")
    require(all(not (BASE/name).exists() for name in targets),"A preparation/build target exists; refuse overwrite")
    # Copy only the main SOURCE. Its DOCX/PDF are newly built after root's edit.
    for name in ("MAIN_PAPER_EN.md",*SUPPLEMENT_FILES,"AUTHOR_INFORMATION.json","review/DOCUMENT_BUILD.json"):
        destination=BASE/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(baseline/name,destination)
        require(sha(destination)==record["files"][name],"Preparation copy changed: "+name)
    shutil.copytree(baseline/"figures",BASE/"figures")
    for name in record["figure_relative_paths"]:require(sha(BASE/name)==record["files"][name],"Figure copy changed")
    inherited_supplement(record)
    manifest.update(status="prepared_for_root_main_edit_no_render",prepared=True,
                    main_source_hash_at_prepare=record["files"]["MAIN_PAPER_EN.md"],
                    inherited_supplement_hashes={name:record["files"][name] for name in SUPPLEMENT_FILES},
                    inherited_supplement_acceptance=manifest["external_dependencies"]["review/FINAL_DOCUMENT_ACCEPTANCE.json"],
                    builder_wrapper_sha256=sha(Path(__file__)))
    save_manifest(manifest);print(json.dumps(manifest,indent=2))


def render_main(replace_render=False):
    baseline,record=verified_baseline();manifest=json.loads((BASE/"PREPARATION_MANIFEST.json").read_text())
    require(manifest["baseline_locks_sha256"]==sha(BASE/"BASE_SOURCE_LOCKS.json"),"Baseline lock manifest changed")
    require(manifest["prepared"] is True,"Run prepare once before editing/building the main")
    require(manifest["main_source_hash_at_prepare"]==record["files"]["MAIN_PAPER_EN.md"],"Preparation source lock changed")
    inherited=inherited_supplement(record)
    outputs=("MAIN_PAPER_RAW.docx","MAIN_PAPER_EN.docx","MAIN_PAPER_EN.pdf")
    require(replace_render or all(not (BASE/name).exists() for name in outputs),
            "Main render output exists; use explicit --replace-render for an intended new render")
    text=(BASE/"MAIN_PAPER_EN.md").read_text();original=(baseline/"MAIN_PAPER_EN.md").read_text()
    require(text.splitlines()[0]==original.splitlines()[0],"Original title must be retained")
    front=text.split("## Abstract",1)[0]
    require("Wang Yang" in front and "High School Affiliated to Shanghai Jiao Tong University" in front
            and "[Email to be completed by the author]" in front,"Confirmed author/school/email placeholder changed")
    require(sha(BASE/"AUTHOR_INFORMATION.json")==record["files"]["AUTHOR_INFORMATION.json"],"Author metadata changed")
    review=BASE/"review";review.mkdir(exist_ok=True)
    diff="".join(difflib.unified_diff(original.splitlines(keepends=True),text.splitlines(keepends=True),
                    fromfile="accepted_two_state/MAIN_PAPER_EN.md",tofile="focused_edition/MAIN_PAPER_EN.md"))
    diff_path=review/"MAIN_SOURCE_REVISION.diff";diff_path.write_text(diff)
    source_hash=sha(BASE/"MAIN_PAPER_EN.md");supp_metadata=json.loads((BASE/"review/DOCUMENT_BUILD.json").read_text())
    old_supp=next(row for row in supp_metadata["documents"] if row["name"]=="SUPPLEMENT")
    attempt={"status":"render_started","source_sha256":source_hash,"source_diff_sha256":sha(diff_path),
             "main_source_changed_from_baseline":source_hash!=manifest["main_source_hash_at_prepare"],
             "replace_render_explicit":replace_render,"builder_wrapper_sha256":sha(Path(__file__)),
             "main_outputs_before":{name:sha(BASE/name) if (BASE/name).is_file() else None for name in outputs}}
    manifest.setdefault("render_attempts",[]).append(attempt);manifest["status"]="main_render_started";save_manifest(manifest)
    previous_argv=sys.argv;previous_path=list(sys.path);previous_bytecode=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        builder=Path(record["reused_builder"]["path"])
        spec=importlib.util.spec_from_file_location("accepted_focused_main_builder",builder)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.BASE=BASE;module.PREVIOUS=baseline;module.FORMATTER=Path(record["reused_formatter"]["path"]).parent
        sys.argv=[str(BASE/"build_focused_documents.py"),"--only","MAIN_PAPER"]
        module.main()
        require(sha(BASE/"MAIN_PAPER_EN.md")==source_hash,"Main source changed during build")
        inherited_supplement(record);verified_baseline()
        result=json.loads((BASE/"review/DOCUMENT_BUILD.json").read_text())
        new_supp=next(row for row in result["documents"] if row["name"]=="SUPPLEMENT")
        require(new_supp==old_supp,"Inherited supplement build/preview metadata changed")
        main=next(row for row in result["documents"] if row["name"]=="MAIN_PAPER")
        for extension,key in ((".md","source_sha256"),(".docx","docx_sha256"),(".pdf","pdf_sha256")):
            require(sha(BASE/("MAIN_PAPER_EN"+extension))==main[key],"Current main artifacts not bound to build")
        require(main["preservation"]["passed"] is True and main["preservation"]["final_layout_preservation_passed"] is True,
                "Main source/text/math/media/table/link preservation failed")
        attempt.update(status="rendered_pending_root_claim_and_actual_visual_review",main_artifacts=main,
                       inherited_supplement_byte_identical=True,inherited_supplement_metadata_identical=True)
        manifest.update(status="main_rendered_pending_root_claim_and_actual_visual_review",main_rendered=True,
                        inherited_supplement_hashes=inherited,main_source_current_sha256=source_hash,
                        main_source_byte_identical_current=source_hash==manifest["main_source_hash_at_prepare"])
    except BaseException as error:
        attempt.update(status="render_failed_preserve_artifacts",error={"type":type(error).__name__,"message":str(error)})
        manifest["status"]="render_failed_preserve_artifacts";raise
    finally:
        sys.argv=previous_argv;sys.path[:]=previous_path;sys.dont_write_bytecode=previous_bytecode;save_manifest(manifest)
    print(json.dumps(manifest,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="operation",required=True)
    sub.add_parser("prepare");render=sub.add_parser("render-main")
    render.add_argument("--replace-render",action="store_true")
    args=parser.parse_args();os.sched_setaffinity(0,{26})
    for name in THREAD_KEYS:os.environ[name]="1"
    os.environ["CUDA_VISIBLE_DEVICES"]="-1";os.environ["PYTHONDONTWRITEBYTECODE"]="1"
    if args.operation=="prepare":prepare()
    else:render_main(args.replace_render)


if __name__=="__main__":main()
