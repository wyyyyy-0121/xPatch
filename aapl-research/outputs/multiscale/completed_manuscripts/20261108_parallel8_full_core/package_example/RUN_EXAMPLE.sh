#!/usr/bin/env bash
set -euo pipefail
cd /home/ayaka/wslcode/aapl-research
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
taskset -c 24,26 /home/ayaka/wslcode/aapl-research/.venv/bin/python -m multiscale.study_package forecast --package /home/ayaka/wslcode/aapl-research/outputs/multiscale/completed_core_evidence/20261108_parallel8_continuation/inference_package --registry-sha256 e0d31d43badf53e58b7b7303b8207ddece847d3d856a60925a9981dee3d8cdee --bars /home/ayaka/wslcode/aapl-research/outputs/multiscale/completed_manuscripts/20261108_parallel8_full_core/package_example/H60_RETURN_PAST_480_BARS.npz --profile minute --horizon 60 --target future_log_return --model xpatch --as-of-ns 1754055060000000000 --history-identity saved-original-fold5-past-only-input:ce663219449f19b8c7eba790ccb4652494d3aa513d677e3595a1fa311ea75deb --device cpu
