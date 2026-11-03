"""Fixed nested-search state machine: no choices depend on outer scores."""
from statistics import median

from .io import ROOT, write_json
from .search import rank_trials
from .splits import inner_partitions


class Experiment:
    def __init__(self, outer, panel, name, search, config, signature, run_directory):
        self.outer, self.panel, self.name = outer, panel, name
        self.search, self.config, self.signature = search, config, signature
        self.identity = f"{outer.name}_{panel}_{name}"
        self.directory = run_directory / self.identity
        self.directory.mkdir(parents=True, exist_ok=True)
        self.inner = inner_partitions(outer, config["splits"]["inner_folds"], config["splits"]["inner_validation_sessions"],
            config["splits"]["calibration_sessions"], config["splits"]["purge_sessions"])
        self.trials = {row["id"]: row for row in search["models"][name]}
        self.stage, self.stage_index = "initial", 0
        self.waiting, self.results = set(), []
        self.current, self.selection = [], None
        self.all_results = []
        self.stages = []

    def _request(self, trial, split, seed, stage, predecessor=None, fixed=None):
        job_id = f"{self.identity}/{stage}/{trial['id']}/{split.name}/s{seed}"
        request = {"job_id": job_id, "directory": str((self.directory / stage / trial["id"] / split.name / f"s{seed}").relative_to(ROOT)),
            "signature": self.signature, "phase": "outer" if stage == "outer" else "inner",
            "partition": split.dictionary(), "model": self.name, "panel": self.panel, "config_id": trial["id"],
            "params": trial["params"], "seed": seed}
        if self.name == "xpatch":
            request["max_epochs"] = self.config["search"]["xpatch"]["halving_epochs"][self.stage_index]
            if predecessor:
                request["resume_directory"] = predecessor["directory"]
            if fixed is not None:
                request["fixed_epochs"] = fixed
        if self.name == "lightgbm" and fixed is not None:
            request["fixed_iterations"] = fixed
        return request

    def _wait(self, requests):
        self.waiting = {r["job_id"] for r in requests}
        self.results = []
        self.current = requests
        return requests

    def initial_requests(self):
        if self.name == "last_rv":
            self.selection = {"config_id": next(iter(self.trials)), "fixed_training_length": None,
                "method": "fixed_baseline_no_search", "hpo_complete": True}
            return self._outer_requests()
        return self._wait([self._request(trial, split, 2026, "initial")
            for trial in self.trials.values() for split in self.inner])

    def accept(self, result):
        if result["job_id"] not in self.waiting:
            raise ValueError("Experiment received an unrelated/duplicate job")
        self.waiting.remove(result["job_id"])
        self.results.append(result)
        self.all_results.append(result)
        return not self.waiting

    def advance(self):
        if self.waiting:
            return []
        if self.stage == "outer":
            status = "complete" if all(row["status"] == "complete" for row in self.results) else "failed"
            self.stage = "done"
            write_json(self.directory / "experiment.json", {"identity": self.identity, "status": status,
                "selection": self.selection, "outer_jobs": [row["directory"] for row in self.results],
                "outer_scores_inspected_for_search": False})
            return []
        seeds = tuple(self.config["search"]["seeds"]) if self.stage == "finalists" else (2026,)
        ranked = rank_trials(self.results, expected_folds=len(self.inner), expected_seeds=seeds)
        self.stages.append({"stage": self.stage, "epoch_stage": self.stage_index, "requested_jobs": len(self.current),
            "completed_jobs_including_reused_seed2026": sum(r["status"] == "complete" for r in self.results), "ranking": ranked})
        if not ranked:
            self.stage = "done"
            write_json(self.directory / "experiment.json", {"identity": self.identity, "status": "failed_no_complete_inner_config",
                "stages": self.stages, "jobs": [r["directory"] for r in self.all_results]})
            return []
        if self.name == "xpatch" and self.stage != "finalists":
            epochs = self.config["search"]["xpatch"]["halving_epochs"]
            if self.stage_index + 1 < len(epochs):
                self.stage_index += 1
                keep = self.config["search"]["xpatch"]["halving_keep"][self.stage_index]
                selected = {r["config_id"] for r in ranked[:keep]}
                prior = {(r["config_id"], r["fold"]): r for r in self.results}
                self.stage = f"halving_e{epochs[self.stage_index]}"
                requests = [self._request(self.trials[identity], split, 2026, self.stage, prior[(identity, split.name)])
                    for identity in sorted(selected) for split in self.inner]
                return self._wait(requests)
        if self.name in self.config["search"]["stochastic_models"] and self.stage != "finalists":
            finalists = {r["config_id"] for r in ranked[:self.config["search"]["finalist_configs"]]}
            self.base_finalists = [r for r in self.results if r["config_id"] in finalists]
            self.stage = "finalists"
            requests = [self._request(self.trials[identity], split, seed, "finalists")
                for identity in sorted(finalists) for seed in self.config["search"]["seeds"][1:] for split in self.inner]
            self._wait(requests)
            # Seed 2026 is reused from its completed full-budget inner evaluation.
            self.results = list(self.base_finalists)
            return requests
        chosen = ranked[0]
        rows = [r for r in self.results if r["config_id"] == chosen["config_id"]]
        fixed = None
        if self.name == "xpatch":
            fixed = max(1, int(median(r["fitting"]["best_epoch"] for r in rows)))
        elif self.name == "lightgbm":
            fixed = max(1, int(median(r["fitting"]["best_iteration"] for r in rows)))
        self.selection = {**chosen, "params": self.trials[chosen["config_id"]]["params"], "fixed_training_length": fixed,
            "method": "three_seed_three_fold_mean" if len(seeds) == 3 else "three_fold_mean",
            "hpo_complete": True, "stages": self.stages,
            "inner_jobs": [r["directory"] for r in self.all_results], "selection_jobs": [r["directory"] for r in rows],
            "selected_before_outer_labels": True, "fixed_length_rule": "median_best_iteration_or_epoch_of_selected_inner_jobs"}
        return self._outer_requests()

    def _outer_requests(self):
        write_json(self.directory / "selection.json", self.selection)
        self.stage = "outer"
        trial = self.trials[self.selection["config_id"]]
        seeds = self.config["search"]["seeds"] if self.name in self.config["search"]["stochastic_models"] else [2026]
        return self._wait([self._request(trial, self.outer, seed, "outer", fixed=self.selection["fixed_training_length"])
            for seed in seeds])
