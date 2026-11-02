# AAPL v2 实验入口

协议见仓库根目录 `PROTOCOL.md`，完整阶段状态见 `EXECUTION_PLAN.md`。v1 的 `finance_ls`、数据和结果保留。v2 主面板为 close-only，次面板为 OHLCV；所有模型使用相同一分钟 origin、240 分钟可得历史和 1/5/12 分钟累计收益标签。半日与缺柱不能制造样本；原始历史数据不代表实时当时可获得的最终版本。

当前环境为 `~/.venvs/xpatch-v2`。下载数据与权重仅存本机，未提交 Git。公开权重 identity 与本地 SHA256 见 `configs/aapl_v2_model_lock.json`；供应商全历史审计见 `data_acquisition/V2_SOURCE_AUDIT.md`。

```bash
python -m pytest tests/test_finance_ls.py tests/test_finance_v2_data.py tests/test_alpaca_acquisition.py tests/test_finance_v2_execution.py -q
python -m finance_v2.run --stage audit
python -m finance_v2.run --stage validate
python -m finance_v2.run --stage verify
python -m finance_v2.run --stage freeze
python -m finance_v2.run --stage confirm
python -m finance_v2.run --stage report
python -m finance_v2.run --stage verify
```

验证结果只有全部文件哈希相符才会恢复；源代码、运行依赖版本、配置、协议、数据或权重清单改变会拒绝复用旧工件。检查点保存模型、尺度和选择轮数。校准尾部与训练和季度评价隔离；累计分位数从累计预测残差取得，不能直接把逐步分位数相加。

每次运行保留精确日期分区、scaler、最终和选择 checkpoint、校准预测、季度预测、资源耗时、指标和逐文件哈希。独立 `verify` 从源 CSV 重建同日连续标签、划分和尺度，复核残差分位数与指标，重新加载监督模型并完整复现预测；预训练模型复现各固定运行的首尾批次。

完整开发期为 10 季度 × 2 面板 × 13 个模型/种子运行，共 260 次。程序先完成一个季度/面板中的全部模型，再进入下一个。验证后依据预声明门槛决定是否开启一次历史确认；确认期 2025H2–2026H1 已被旧项目使用，不能称为新盲测。最终仍须冻结之后的新日期前瞻观察。预测门槛失败时保存负结果，不启动交易研究。

`benchmark` 在首次验证季前的历史数据上测量资源，并检查未来数据改写不改变推理。公开模型可通过 `python -m finance_v2.prepare_models` 在固定 revision 下载；境内网络如需镜像可显式传 `--endpoint https://hf-mirror.com`，传输端点与镜像证据限制会写入清单。本流程没有付款、上传凭据或交易下单。
