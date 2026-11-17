#!/usr/bin/env bash
set -euo pipefail
export AAPL_CPU_IDS=24,26 AAPL_CPU_THREADS=2 CUDA_VISIBLE_DEVICES=-1
bash /home/ayaka/wslcode/aapl-research/scripts/python.sh -m multiscale.study_package forecast --package /home/ayaka/wslcode/aapl-research/outputs/multiscale/historical_inference_packages/20261108_all20_tasks --registry-sha256 9727a673f69e0e1f13ed54027aff328fd4dfc03a841595b60215fc5ef95746aa --bars /home/ayaka/wslcode/aapl-research/outputs/multiscale/historical_inference_packages/20261108_all20_tasks_audit/example/minute_h60_past_only_input.npz --profile minute --horizon 60 --target future_log_return --model xpatch --as-of-ns 1754055060000000000 --history-identity alpaca_admission_f6fc738205532cf5016d74b2baa4691692cb03674c3db9b6a8ccf8ce8126ae86_origin_446670
