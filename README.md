<div align="center">
  <h2><b> (AAAI25) xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition </b></h2>
</div>

<div align="center">

[![](https://img.shields.io/badge/arXiv:2412.17323-B31B1B?logo=arxiv)](https://arxiv.org/pdf/2412.17323)
[![](https://img.shields.io/badge/AAAI'25-xPatch-deepgreen)](https://ojs.aaai.org/index.php/AAAI/article/view/34270)
![](https://img.shields.io/github/last-commit/stitsyuk/xPatch)
![](https://badges.pufler.dev/visits/stitsyuk/xPatch)
![](https://img.shields.io/github/stars/stitsyuk/xPatch)
![](https://img.shields.io/github/forks/stitsyuk/xPatch)

</div>

This is an official implementation of [xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition](https://arxiv.org/pdf/2412.17323).

## AAPL minute forecasting

The active delivery project is the [RV30 research and inference tool](delivery/README.md), with its [final acceptance goal](GOAL.md) and [current status](delivery/status.json). It predicts thirty-minute realized variance from minute origins; it does not claim price direction or trading profitability. New code uses the stable `finance/` entry point. The existing v1/v2 experiments below remain sealed historical evidence.

### Sealed twelve-minute return experiment (v2)

The sealed research entrypoint is [finance_v2](finance_v2/README.md). The [execution ledger](EXECUTION_PLAN.md) records its fixed stage order and evidence; the [protocol](PROTOCOL.md) fixes its information sets, ten quarterly walk-forward folds, calibration, statistical gates, and resource budget. The [literature review](research/AAPL_MINUTE_REVIEW.md) distinguishes finance journal evidence from model benchmarks and author technical reports.

Verified Alpaca SIP/raw history covers 2021-01-04 through 2026-06-30: 1,378 sessions and 535,620 regular-session minute bars, with 100% per-session calendar coverage. See the [data-source audit](data_acquisition/V2_SOURCE_AUDIT.md). The comparison includes zero returns, Ridge, LightGBM, original xPatch48/240, Kronos-small, and Chronos-2, with immutable pretrained [checkpoint identities](configs/aapl_v2_model_lock.json).

Development validation is complete: all 260 runs passed two artifact and prediction verification rounds. Every candidate had higher pooled 12-minute MSE than zero returns, so neither panel passed the predeclared development gate; historical confirmation and trading research were skipped. See the [results and limitations](finance_v2/RESULTS.md) and [machine-readable summary](finance_v2/results_summary.json). The 240-minute same-session context limits evaluated origins to afternoon trading. Previously exposed historical dates are not a pristine holdout, and historical final bars do not establish publication-time availability or trading profitability. Credentials, market data, model weights, and generated experiment artifacts remain local.

## Financial long/short-stream experiments

This fork adds **xPatch-LS**: a 240-minute trend stream and a 48-minute residual stream for the next 12 minute log returns. See the [experiment guide](finance_ls/README.md) for the architecture, chronological validation, frozen held-out evaluation, baselines, ablations, and independent artifact verification. Run from the repository root:

This v1 experiment is complete: the primary long/short proposal did not establish stable improvement over original xPatch under its protocol. Retain its [results](finance_ls/RESULTS.md) as a historical record; use the v2 entrypoint for the new data and comparison.

```bash
python -m pip install -r requirements-finance.txt
python -m pytest tests/test_finance_ls.py -q
python -m finance_ls.run --stage all
python -m finance_ls.verify
```

The financial entrypoint is separate from the older `appl/Min-by-Min` scripts. Research results are generated in `artifacts/finance_ls/REPORT.md`; predictive improvements must be established by those experiments.

## Model Overview

E**x**ponential **Patch** (xPatch) is a novel dual-stream architecture that utilizes exponential decomposition.

### Seasonal-Trend Decomposition

**Simple Moving Average (SMA)** is the unweighted mean of the previous data points.

<p align="center">
<img src="./figures/sma.png" alt="" style="width: 80%;" align=center />
</p>

**Exponential Moving Average (EMA)** is an exponential smoothing method that assigns greater weight to more recent data points while smoothing out older data.

<p align="center">
<img src="./figures/ema.png" alt="" style="width: 80%;" align=center />
</p>

### Architecture

Dual-flow architecture consists of an MLP-based linear stream and a CNN-based non-linear stream.

<p align="center">
<img src="./figures/xpatch.png" alt="" align=center />
</p>

## Results

### Long-term Forecasting with Unified Experimental Settings

In the unified experimental settings, xPatch achieves the best averaged performance on 60% of the datasets using the MSE metric and 70% of the datasets using the MAE metric.

<p align="center">
<img src="./figures/hyper-unified.png" alt="" style="width: 80%;" align=center />
</p>

### Long-term Forecasting with Hyperparameter Search

In the hyperparameter search settings, xPatch achieves the best averaged performance on 70% of the datasets using the MSE metric and 90% of the datasets using the MAE metric.

<p align="center">
<img src="./figures/hyper-search.png" alt="" style="width: 80%;" align=center />
</p>

### Efficiency on Long Look-back Windows

We explore the ability of different models to learn from a longer lookback window.

<p align="center">
<img src="./figures/lookback.png" alt="" style="width: 80%;" align=center />
</p>

### Dual Flow Net

We explore the impact of the dual flow network in xPatch architecture and assess the contribution of each stream. The four possible configurations:
- Original: Seasonality -> non-linear stream, Trend -> linear stream
- Reversed: Seasonality -> linear stream, Trend -> non-linear stream
- Non-linear only: Seasonality -> non-linear stream, Trend -> non-linear stream
- Linear only: Seasonality -> linear stream, Trend -> linear stream

<p align="center">
<img src="./figures/dual-flow.png" alt="" style="width: 80%;" align=center />
</p>

## "Drop-last" trick

Recent models have widely adopted the "drop-last" trick, which is well explained in the [TFB](https://www.vldb.org/pvldb/vol17/p2363-hu.pdf) paper. Since re-running all experiments for existing benchmark models without this trick would be highly complex and time-consuming, we chose to rely on the benchmark results reported in their official papers to ensure fair experimentation. These reported results incorporate all tricks and hyperparameter searches applied by the original authors. Our primary objective was to use these published results (even if they are not entirely fair) rather than re-conducting the benchmark experiments ourselves.

However, the recent work [TFB](https://www.vldb.org/pvldb/vol17/p2363-hu.pdf) introduces a new and fair benchmark that excludes the "drop-last" trick. The authors fairly re-evaluated all existing models, including a comprehensive hyperparameter search. Therefore, we incorporate the results of xPatch fair experiments conducted without the "drop-last" trick by referencing the benchmark results from the new [OpenTS](https://decisionintelligence.github.io/OpenTS/leaderboards/multivariate_time_series) leaderboard.

Notably, xPatch achieves state-of-the-art performance on the longest and most challenging datasets — Weather, Traffic, and Electricity — across all prediction lengths.

<p align="center">
<img src="./figures/hyper-fair.png" alt="" style="width: 80%;" align=center />
</p>

## Getting Started

1. Install conda environment: ```conda env create -f environment.yml```

2. Download data. You can download the datasets from [Google Driver](https://drive.google.com/u/0/uc?id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP&export=download), [Baidu Driver](https://pan.baidu.com/share/init?surl=r3KhGd0Q9PJIUZdfEYoymg&pwd=i9iy) or [Kaggle Datasets](https://www.kaggle.com/datasets/wentixiaogege/time-series-dataset). All datasets are pre-processed and can be used easily. Create a seperate folder ```./dataset``` and put all the files in the directory.

3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. The script for unified settings is *xPatch_unified*, the script for hyperparameter search is *xPatch_search*, while for fair experiment without the "drop-last" trick you can use *xPatch_fair*. You can reproduce the experiments by:

```
bash scripts/xPatch_search.sh
```

All experiments were conducted on a single Quadro RTX 6000 GPU. You can adjust the hyperparameters based on your needs (e.g. batch size, patch length, lookback windows and prediction lengths, alpha parameter for exponential decomposition). We also provide code for the baseline models and for ablation experiments from Appendix (EMA decomposition, arctangent loss, sigmoid learning rate adjustment scheme, inference time) in ```./ablation```.

## Acknowledgement

We appreciate the following github repos for their valuable code and effort:
- Autoformer (https://github.com/thuml/Autoformer)
- FEDformer (https://github.com/MAZiqing/FEDformer)
- ETSformer (https://github.com/salesforce/ETSformer)
- DLinear (https://github.com/cure-lab/LTSF-Linear)
- RLinear (https://github.com/plumprc/RTSF)
- PatchTST (https://github.com/yuqinie98/PatchTST)
- CARD (https://github.com/wxie9/CARD)
- TimeMixer (https://github.com/kwuking/TimeMixer)
- iTransformer (https://github.com/thuml/iTransformer)
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- RevIN (https://github.com/ts-kim/RevIN)
- TFB (https://github.com/decisionintelligence/TFB)

## Contact

If you have any questions or concerns, please contact us at stitsyuk@kaist.ac.kr or submit an issue.

## Citation

If you find this repository useful in your research, please consider citing our paper as follows:

```
@inproceedings{stitsyuk2025xpatch,
  title={xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition},
  author={Stitsyuk, Artyom and Choi, Jaesik},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={19},
  pages={20601--20609},
  year={2025}
}
```
