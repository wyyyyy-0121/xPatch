"""Thin source-only wrapper: prepare accepted S38, then render selected documents.

No market array or numerical library imports. This source has not been run.
The root must supply the independently accepted screen and final English S38.
"""
from pathlib import Path
import argparse
import difflib
import hashlib
import importlib.util
import json
import re
import shutil
import sys

BASE=Path(__file__).resolve().parent
MAIN_FILES=("MAIN_PAPER_EN.md","MAIN_PAPER_EN.docx","MAIN_PAPER_EN.pdf")


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locks():
    record=json.loads((BASE/"BASE_SOURCE_LOCKS.json").read_text())
    accepted=Path(record["accepted_directory"]).resolve()
    require(BASE!=accepted and not BASE.is_relative_to(accepted) and not accepted.is_relative_to(BASE),
            "New edition must be separate from the accepted directory")
    for name in (*MAIN_FILES,"MAIN_PAPER_RAW.docx","SUPPLEMENT_EN.md","SUPPLEMENT_EN.docx",
                 "SUPPLEMENT_EN.pdf","SUPPLEMENT_RAW.docx","review","figures"):
        require((BASE/name).resolve().is_relative_to(BASE),"Output symlink escapes new edition: "+name)
    for name,digest in record["files"].items():
        require(sha(accepted/name)==digest,"Accepted baseline changed: "+name)
    return accepted,record


def copy_once(source,destination):
    require(not destination.exists(),"Refuse to overwrite: "+str(destination))
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,destination)
    require(sha(source)==sha(destination),"Document copy changed")


def accepted_screen(directory):
    paths={"complete":directory/"COMPLETE_SCREEN_SUPERVISOR.json",
           "results":directory/"postprocessing/RESULTS.json",
           "review":directory/"postprocessing/INDEPENDENT_REVIEW.json"}
    docs={name:json.loads(path.read_text()) for name,path in paths.items()}
    complete,results,review=(docs[key] for key in ("complete","results","review"))
    require(complete["screen_complete"] is True and review["status"]=="PASS"
            and results["status"]=="main_scoring_complete_independent_review_pending",
            "Complete independent screen acceptance is required before writing S38")
    require(len({doc["protocol_sha256"] for doc in docs.values()})==1,"Scientific receipts disagree on protocol")
    require(type(review["screen_passed"]) is bool and type(results["screen_conjunction_passed"]) is bool
            and review["screen_passed"]==results["screen_conjunction_passed"],"Scientific gate mismatch")
    require(complete["finished_epoch"]<=complete["deadline_epoch"]
            and review["finished_epoch"]<=complete["deadline_epoch"]
            and results["finished_epoch"]<=complete["deadline_epoch"],"Original acceptance deadline exceeded")
    require(all(doc["research_goal_achieved"] is False for doc in docs.values()),"Screen may not imply Goal success")
    return paths,docs


def prepare(args):
    accepted,record=locks()
    require(args.screen_directory.is_absolute() and args.s38_file.is_absolute(),"Explicit absolute inputs required")
    paths,docs=accepted_screen(args.screen_directory)
    section=args.s38_file.read_bytes()
    require(section.startswith(b"# S38.") and b"DRAFT OUTLINE ONLY" not in section,
            "Root-reviewed final S38, not the outline, is required")
    supplement=(accepted/"SUPPLEMENT_EN.md").read_bytes()
    require(b"# S38." not in supplement,"Baseline already contains S38")
    require(not (BASE/"SUPPLEMENT_EN.md").exists(),"Refuse repeated preparation")
    for name in ("MAIN_PAPER_EN.md","MAIN_PAPER_EN.docx","MAIN_PAPER_EN.pdf","AUTHOR_INFORMATION.json"):
        copy_once(accepted/name,BASE/name)
    copy_once(accepted/"review/DOCUMENT_BUILD.json",BASE/"review/DOCUMENT_BUILD.json")
    (BASE/"SUPPLEMENT_EN.md").write_bytes(supplement+b"\n\n"+section)
    require((BASE/"SUPPLEMENT_EN.md").read_bytes().startswith(supplement),"Old supplement prefix changed")
    evidence={}
    for name,path in paths.items():
        target=BASE/"evidence/two_state"/path.name;copy_once(path,target)
        evidence[name]={"source_path":str(path),"snapshot":str(target.relative_to(BASE)),"sha256":sha(path)}
    manifest={"status":"sources_prepared_pending_scalar_claim_render_and_visual_review",
              "accepted_base":str(accepted),"protocol_sha256":docs["complete"]["protocol_sha256"],
              "old_supplement_prefix_sha256":record["files"]["SUPPLEMENT_EN.md"],
              "S38_source_sha256":sha(args.s38_file),"source_receipts":evidence,
              "builder_wrapper_sha256":sha(Path(__file__)),
              "accepted_builder_sha256":record["files"]["build_integrated_documents.py"],
              "main_lock_scope":"preparation_default_only_final_main_claim_review_required",
              "main_three_files_byte_identical_at_prepare":True,
              "main_artifact_hashes_at_prepare":{name:record["files"][name] for name in MAIN_FILES},
              "main_three_files_byte_identical_current":True,"market_arrays_loaded":False,
              "method_success":False,"research_goal_achieved":False}
    (BASE/"SOURCE_CHANGELOG.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))


def install_new_edition_table_layout(module):
    """Apply only S38 layout fixes after the unchanged accepted layout routine."""
    accepted_layout=module.table_layout
    def local_layout(path,double):
        require(module.BASE.resolve()==BASE,"Local layout may only target the new edition")
        require(Path(path).resolve().is_relative_to(BASE),"Layout output escapes the new edition")
        before=module.inspect(path)
        changed=accepted_layout(path,double)
        if not double:
            doc=module.Document(path)
            caption=re.compile(r"Table\s+S38\.\d+")
            for paragraph in doc.paragraphs:
                if caption.match(paragraph.text):
                    paragraph.paragraph_format.keep_with_next=True
                    paragraph.paragraph_format.keep_together=True
            # Bind only tables following new S38 captions, preserving every
            # existing table's pagination and all preceding supplement pages.
            s38_tables=set();pending=False;table_index=0
            for element in doc.element.body:
                if element.tag.endswith("}p"):
                    text="".join(node.text or "" for node in element.iter() if node.tag.endswith("}t"))
                    if caption.match(text):pending=True
                elif element.tag.endswith("}tbl"):
                    if pending:s38_tables.add(table_index)
                    pending=False;table_index+=1
            headers=["Task","Gain (%)","Short-block 95% interval","Five-session 95% interval","Passed"]
            ratios=[.18,.14,.29,.29,.10]
            for index,table in enumerate(doc.tables):
                heads=[cell.text for cell in table.rows[0].cells]
                if heads==headers:
                    for column,ratio in zip(table.columns,ratios):column.width=module.Mm(178*ratio)
                    for row in table.rows:
                        for cell,ratio in zip(row.cells,ratios):cell.width=module.Mm(178*ratio)
                    changed.append(heads)
                if index in s38_tables:
                    for cell in table.rows[0].cells:
                        for paragraph in cell.paragraphs:
                            paragraph.paragraph_format.keep_with_next=True
                            paragraph.paragraph_format.keep_together=True
            doc.save(path)
        require(module.inspect(path)==before,"S38 layout changed text, math, media, tables or links")
        return changed
    module.table_layout=local_layout


def render_documents(scope):
    """Rebuild only the requested scope, auditing any new-main source revision."""
    require(scope in ("SUPPLEMENT","MAIN_PAPER","both"),"Unknown render scope")
    accepted,record=locks();changelog_path=BASE/"SOURCE_CHANGELOG.json"
    require(changelog_path.is_file(),"Prepare qualified final sources first")
    changelog=json.loads(changelog_path.read_text())
    baseline={name:record["files"][name] for name in MAIN_FILES}
    # Migrate the earlier skeleton's preparation-only claim without carrying an
    # ambiguous byte-identical assertion forward into a revised final main.
    previous_claim=changelog.pop("main_three_files_byte_identical",None)
    if "main_three_files_byte_identical_at_prepare" not in changelog:
        require(previous_claim is True,"Missing original preparation main lock")
        changelog["main_three_files_byte_identical_at_prepare"]=True
    require(changelog["main_three_files_byte_identical_at_prepare"] is True,"Preparation lock lost")
    changelog.setdefault("main_artifact_hashes_at_prepare",baseline)
    require(changelog["main_artifact_hashes_at_prepare"]==baseline,"Preparation baseline hashes changed")
    old_supplement=(accepted/"SUPPLEMENT_EN.md").read_bytes()
    require((BASE/"SUPPLEMENT_EN.md").read_bytes().startswith(old_supplement),"Accepted S1-S37 prefix changed")
    original_main=(accepted/"MAIN_PAPER_EN.md").read_text()
    current_main=(BASE/"MAIN_PAPER_EN.md").read_text()
    require(current_main.splitlines()[0]==original_main.splitlines()[0],"Original paper title changed")
    before={name:sha(BASE/name) for name in MAIN_FILES}
    review=BASE/"review";review.mkdir(exist_ok=True)
    diff="".join(difflib.unified_diff(original_main.splitlines(keepends=True),current_main.splitlines(keepends=True),
                                   fromfile="accepted_S1_S37/MAIN_PAPER_EN.md",tofile="new_edition/MAIN_PAPER_EN.md"))
    diff_path=review/"MAIN_SOURCE_REVISION.diff";diff_path.write_text(diff)
    audit={"status":"render_started_pending_completion","render_scope":scope,
           "main_artifact_hashes_at_prepare":baseline,"main_artifact_hashes_before_render":before,
           "main_source_changed_from_prepare":before["MAIN_PAPER_EN.md"]!=baseline["MAIN_PAPER_EN.md"],
           "main_source_diff_path":str(diff_path.relative_to(BASE)),"main_source_diff_sha256":sha(diff_path),
           "old_S1_S37_bytes_prefix_preserved":True,"accepted_directory_overwritten":False,
           "builder_wrapper_sha256":sha(Path(__file__)),"method_success":False,"research_goal_achieved":False}
    audit_path=review/"MAIN_REVISION_AUDIT.json"
    audit_path.write_text(json.dumps(audit,indent=2)+"\n")
    changelog.update(status="render_started_pending_completion",main_current_hashes=before,
                     main_three_files_byte_identical_current=before==baseline,
                     main_revision_audit=str(audit_path.relative_to(BASE)))
    changelog_path.write_text(json.dumps(changelog,indent=2)+"\n")
    previous_argv=sys.argv;previous_path=list(sys.path);previous_bytecode=sys.dont_write_bytecode
    try:
        # Importing the accepted formatter must not create __pycache__ there.
        sys.dont_write_bytecode=True
        spec=importlib.util.spec_from_file_location("accepted_S37_document_builder",accepted/"build_integrated_documents.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.BASE=BASE;module.PREVIOUS=accepted;module.FORMATTER=accepted/"code"
        install_new_edition_table_layout(module)
        sys.argv=[str(BASE/"build_two_state_documents.py"),"--only",scope]
        module.main()
        require(sha(BASE/"MAIN_PAPER_EN.md")==before["MAIN_PAPER_EN.md"],"Build changed main source")
        require((BASE/"SUPPLEMENT_EN.md").read_bytes().startswith(old_supplement),"Build changed old supplement prefix")
        if scope=="SUPPLEMENT":
            require({name:sha(BASE/name) for name in MAIN_FILES}==before,"Supplement build changed main artifact")
        locks()  # Every accepted-baseline hash must still match after rendering.
        build=json.loads((review/"DOCUMENT_BUILD.json").read_text())
        bindings={}
        for item in build["documents"]:
            name=item["name"]
            bindings[name]=all(sha(BASE/(name+"_EN"+extension))==item[key]
                              for extension,key in ((".md","source_sha256"),(".docx","docx_sha256"),(".pdf","pdf_sha256")))
        expected=("MAIN_PAPER","SUPPLEMENT") if scope=="both" else (scope,)
        require(all(bindings.get(name) is True for name in expected),"Rendered scope is not bound to its current files")
        after={name:sha(BASE/name) for name in MAIN_FILES}
        unchanged=after==baseline
        audit.update(status="rendered_pending_claim_and_visual_review",main_artifact_hashes_after_render=after,
                     main_three_files_byte_identical_current=unchanged,current_build_file_bindings=bindings,
                     other_document_render_pending=any(not value for value in bindings.values()),
                     accepted_baseline_hashes_still_valid=True)
        changelog.update(status="rendered_pending_scalar_claim_and_visual_review",
                         main_three_files_byte_identical_current=unchanged,main_current_hashes=after,
                         main_revision_audit=str(audit_path.relative_to(BASE)),render_scope=scope,
                         current_build_file_bindings=bindings)
        changelog_path.write_text(json.dumps(changelog,indent=2)+"\n")
    except BaseException as error:
        after={name:sha(BASE/name) if (BASE/name).is_file() else None for name in MAIN_FILES}
        audit.update(status="render_failed_preserve_artifacts",main_artifact_hashes_after_render=after,
                     main_three_files_byte_identical_current=after==baseline,
                     error={"type":type(error).__name__,"message":str(error)})
        changelog.update(status="render_failed_preserve_artifacts",main_current_hashes=after,
                         main_three_files_byte_identical_current=after==baseline)
        changelog_path.write_text(json.dumps(changelog,indent=2)+"\n")
        raise
    finally:
        sys.argv=previous_argv;sys.path[:]=previous_path;sys.dont_write_bytecode=previous_bytecode
        audit_path.write_text(json.dumps(audit,indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="operation",required=True)
    command=sub.add_parser("prepare");command.add_argument("--screen-directory",type=Path,required=True)
    command.add_argument("--s38-file",type=Path,required=True)
    scopes={"render-supplement":"SUPPLEMENT","render-main":"MAIN_PAPER","render-both":"both"}
    for operation in scopes:sub.add_parser(operation)
    args=parser.parse_args()
    if args.operation=="prepare":prepare(args)
    else:render_documents(scopes[args.operation])


if __name__=="__main__":main()
