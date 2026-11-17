"""Re-run the supplied past-only forecast; no training is performed."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
example = root / "manuscript/package_example"
receipt = json.loads((example / "EXAMPLE_VERIFICATION.json").read_text())
history = example / "H60_RETURN_PAST_480_BARS.npz"
assert hashlib.sha256(history.read_bytes()).hexdigest() == receipt["input_sha256"]
registry_sha = (root / "INFERENCE_REGISTRY_SHA256.txt").read_text().strip()
environment = dict(os.environ, PYTHONPATH=str(root / "source"), CUDA_VISIBLE_DEVICES="-1",
                   OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
command = [sys.executable, "-m", "multiscale.study_package", "forecast",
           "--package", str(root / "inference_package"), "--registry-sha256", registry_sha,
           "--bars", str(history), "--profile", "minute", "--horizon", "60",
           "--target", "future_log_return", "--model", "xpatch",
           "--as-of-ns", str(receipt["decision_ns"]),
           "--history-identity", "supplied-past-only-example:" + receipt["input_sha256"], "--device", "cpu"]
subprocess.run(command, cwd=root / "source", env=environment, check=True)
