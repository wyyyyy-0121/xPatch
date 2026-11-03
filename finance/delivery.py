"""Finish dependent stages in order and leave exact actionable delivery status."""
from .acceptance import accept
from .confirmation import confirm
from .evaluation import evaluate
from .inference import freeze_baseline, infer
from .io import ROOT, write_json
from .report import report
from .verify import verify_run


def finish(config, directory):
    evaluation = evaluate(directory, config)
    if evaluation["errors"] or evaluation["complete_model_panel_results"] != 12:
        return {"passed": False, "stage": "development_evaluation_incomplete"}

    def verify_scope(scope):
        for number in (1, 2):
            print(f"Starting {scope} independent verification {number}/2", flush=True)
            verification = verify_run(config, directory, number, scope)
            if verification["passed"]:
                continue
            write_json(ROOT / "delivery/status.json", {"goal_status": "active", "stage": f"{scope}_verification_{number}_failed",
                "run_directory": str(directory.relative_to(ROOT)), "final_acceptance": False})
            return {"passed": False, "stage": f"{scope}_verification_{number}_failed"}

    failure = verify_scope("development")
    if failure:
        return failure
    confirmation = confirm(config, directory)
    if confirmation.get("errors"):
        return {"passed": False, "stage": "historical_confirmation_incomplete"}
    if confirmation.get("job_directories"):
        failure = verify_scope("confirmation")
        if failure:
            return failure
    report(config, directory)
    if confirmation["passed_candidates"]:
        status = {"goal_status": "active", "stage": "waiting_prospective_admission",
            "run_directory": str(directory.relative_to(ROOT)), "passed_historical_candidates": confirmation["passed_candidates"],
            "required": ["resolve_timely_SIP_permission", "freeze_prospective_issuance_and_verifier", "60_complete_on_time_sessions"],
            "final_acceptance": False, "external_items": "delivery/external_dependencies.json"}
        write_json(ROOT / "delivery/status.json", status)
        return {"passed": False, **status}
    freeze_baseline(config, directory)
    failure = verify_scope("delivery")
    if failure:
        return failure
    example = infer(config)
    write_json(ROOT / "delivery/example_prediction.json", example)
    acceptance = accept(config, directory)
    status = {"goal_status": "active", "stage": "engineering_accepted_pending_final_local_commit" if acceptance["passed"] else "final_acceptance_failed",
        "run_directory": str(directory.relative_to(ROOT)), "final_acceptance": acceptance["passed"],
        "outcome": "completed_no_predictive_edge" if acceptance["passed"] else None}
    write_json(ROOT / "delivery/status.json", status)
    return acceptance
