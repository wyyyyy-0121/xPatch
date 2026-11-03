"""The stable command entry point; resource limits precede numerical imports."""
import argparse
import json

from .io import ROOT, read_json, write_json
from .runtime import configure, exclusive_run


def main():
    parser = argparse.ArgumentParser(description="AAPL RV30 research and delivery")
    parser.add_argument("--config", default="configs/rv30.json")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Verify audited source fingerprints and prepare immutable mmap inputs")
    subparsers.add_parser("audit", help="Rebuild exchange-calendar coverage from the fingerprinted source CSVs")
    search = subparsers.add_parser("search-list", help="Generate full prospective parameter list without model scoring")
    search.add_argument("--output", default="artifacts/rv30/setup/search_draft.json")
    smoke = subparsers.add_parser("smoke", help="Run all adapters using pre-2023 data only")
    smoke.add_argument("--run-name")
    freeze = subparsers.add_parser("freeze-protocol", help="Sign tested source, data, model/dependency locks and runtime budgets")
    freeze.add_argument("--smoke-directory", required=True)
    freeze.add_argument("--tests-log", required=True)
    for command in ("train", "deliver"):
        training = subparsers.add_parser(command, help="Run/resume all frozen searches" if command == "train" else "Run the frozen experiment and its conditional delivery stages")
        training.add_argument("--run-name", default="development")
    for command in ("evaluate", "confirm", "verify", "freeze-model", "report", "accept", "finalize"):
        action = subparsers.add_parser(command)
        action.add_argument("--run-directory", required=True)
        if command == "verify":
            action.add_argument("--round", type=int, choices=(1, 2), required=True)
            action.add_argument("--scope", choices=("development", "confirmation", "delivery"), default="development")
    inference = subparsers.add_parser("infer", help="Historical RV replay or explicit live-unavailable status")
    inference.add_argument("--model-manifest", default="delivery/model.json")
    inference.add_argument("--as-of")
    inference.add_argument("--source-manifest")
    inference.add_argument("--mode", choices=("historical", "live"), default="historical")
    inference.add_argument("--output")
    prospective = subparsers.add_parser("prospective", help="Freeze confirmed models, admit actual SIP, collect or verify forward evidence")
    prospective.add_argument("action", choices=("freeze", "admit", "observe", "status", "verify"))
    prospective.add_argument("--run-directory")
    prospective.add_argument("--round", type=int, choices=(1, 2))
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    # Keep all 22 IDs available for spawning disjoint workers; main BLAS remains 1.
    configure(4 if args.command in ("smoke", "infer", "deliver", "finalize", "prospective") else 1, nice_increment=config["runtime"]["nice_increment"])
    if args.command == "prospective" and args.action == "status":
        from .prospective import status
        print(json.dumps(status(config)))
        return 0
    with exclusive_run(ROOT / "artifacts/rv30/driver.lock"):
        if args.command == "prospective":
            from . import prospective
            if args.action == "freeze":
                if not args.run_directory:
                    parser.error("prospective freeze requires --run-directory")
                result = prospective.freeze_plan(config, ROOT/args.run_directory)
                print(json.dumps({"plan_signature":result["signature"],"status":"frozen_pending_actual_SIP_and_60_complete_sessions"}))
                return 0
            if args.action == "admit":
                result = prospective.admission(config)
            elif args.action == "observe":
                result = prospective.observe(config)
            else:
                if args.round is None:
                    parser.error("prospective verify requires --round")
                result = prospective.verify_observation(config,args.round)
            print(json.dumps(result))
            return 0 if result.get("passed",False) else 2
        elif args.command == "prepare":
            from .cache import prepare_data
            directory, identity = prepare_data(config)
            print(json.dumps({"directory": str(directory), "identity": identity}))
        elif args.command == "audit":
            from .data import RVData
            data = RVData.from_manifest(config["data_manifest"])
            print(json.dumps({"status": "passed_for_retrospective_research", "sessions": len(data.days),
                "start": str(data.days[0]), "end": str(data.days[-1]), "coverage": data.regular_audits,
                "source_audit": "data_acquisition/DELIVERY_SOURCE_AUDIT.md", "point_in_time_qualified": False}))
        elif args.command == "search-list":
            from .search import configurations
            write_json(ROOT / args.output, configurations(config))
            print(str(ROOT / args.output))
        elif args.command == "smoke":
            from .smoke import smoke
            result = smoke(config, args.run_name)
            return 0 if result["passed"] else 1
        elif args.command == "freeze-protocol":
            from .provenance import freeze
            result = freeze(config, args.smoke_directory, args.tests_log)
            print(json.dumps({"signature": result["signature"], "status": "frozen"}))
        elif args.command in ("train", "deliver"):
            from .pipeline import run
            from .provenance import verify_freeze
            freeze = verify_freeze(config)
            result = run(config, freeze["signature"], args.run_name)
            if args.command == "deliver":
                from .delivery import finish
                result = finish(config, ROOT / result["directory"])
            print(json.dumps(result))
            return 0 if result.get("passed", True) else 1
        elif args.command == "infer":
            from .inference import infer
            result = infer(config, args.model_manifest, args.as_of, args.mode, args.source_manifest)
            if args.output:
                write_json(ROOT / args.output, result)
            print(json.dumps(result))
            return 0 if result["rv_mean"] is not None else 2
        else:
            directory = ROOT / args.run_directory
            if args.command == "evaluate":
                from .evaluation import evaluate
                result = evaluate(directory, config)
            elif args.command == "confirm":
                from .confirmation import confirm
                result = confirm(config, directory)
            elif args.command == "verify":
                from .verify import verify_run
                result = verify_run(config, directory, args.round, args.scope)
            elif args.command == "finalize":
                from .delivery import finish
                from .provenance import verify_freeze
                verify_freeze(config)
                result = finish(config, directory)
            elif args.command == "freeze-model":
                from .inference import freeze_baseline
                result = freeze_baseline(config, directory)
            elif args.command == "report":
                from .report import report
                result = {"report": str(report(config, directory))}
            elif args.command == "accept":
                from .acceptance import accept
                result = accept(config, directory)
            print(json.dumps(result))
            if args.command in ("evaluate", "confirm") and result.get("errors"):
                return 1
            return 0 if result.get("passed", True) else 1
    return 0
