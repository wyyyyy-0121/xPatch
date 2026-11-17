"""Run the unchanged frozen verifier with a NumPy scalar JSON output adapter.

This fixes only the final report serialization. Original source, formulas,
tolerances, result paths, protocol identity and absolute deadline stay unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--repair-authorization", required=True)
args = parser.parse_args()
authorization = Path(args.repair_authorization).resolve()
a = json.loads(authorization.read_text())
assert a["repair_scope"] == "JSON_output_numpy_float32_scalar_only"
assert digest(__file__) == a["adapter_sha256"]
verifier = Path(a["original_verifier"]).resolve()
protocol = Path(a["original_protocol"]).resolve()
run = Path(a["original_run_directory"]).resolve()
assert digest(verifier) == a["original_verifier_sha256"]
assert digest(protocol) == a["original_protocol_sha256"]
p = json.loads(protocol.read_text())
assert p["stage_deadline_epoch"] == a["original_stage_deadline_epoch"]
assert time.time() < p["stage_deadline_epoch"] - 10
assert not (run / "INDEPENDENT_REPLAY.json").exists()

original_default = json.JSONEncoder.default
conversion_count = 0


def scalar_default(self, value):
    global conversion_count
    numpy_module = sys.modules.get("numpy")
    if numpy_module is not None and type(value) is numpy_module.float32:
        conversion_count += 1
        return float(value)
    return original_default(self, value)


json.JSONEncoder.default = scalar_default
sys.dont_write_bytecode = True
sys.argv = [str(verifier), "--protocol", str(protocol), "--run-directory", str(run)]
namespace = {"__name__": "__main__", "__file__": str(verifier), "__package__": None}
try:
    exec(compile(verifier.read_text(), str(verifier), "exec"), namespace)
finally:
    print(json.dumps({"JSON_scalar_adapter": "numpy.float32_to_Python_float_only",
                      "conversion_count": conversion_count,
                      "numerical_code_modified": False}), flush=True)
