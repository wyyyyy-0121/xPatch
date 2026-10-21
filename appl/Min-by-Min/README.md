# AAPL 多时间尺度预测

本目录实现年度、季度、月度、小时和分钟预测，同时比较未来收盘价与相对预测起点的未来对数收益率。模型包括最后收盘价、漂移、季节性、Ridge、MLP、GRU、LSTM 和 xPatch。不做交易成本策略回测，不承诺预测收益。

## 安装与数据

```bash
cd appl/Min-by-Min
python -m venv .venv
source .venv/bin/activate
# 为 RTX 5070 Ti 安装支持 Blackwell 的 CUDA PyTorch；不要使用不支持该卡的旧版。
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python download_data.py --symbol AAPL --interval 1d
python download_data.py --symbol AAPL --interval 1mo
```

下载使用公开 Yahoo chart API，保存完整可得范围、来源网址、下载时间、时区、复权说明、SHA-256 及下载凭证。默认复用已有缓存，`--refresh` 重新下载；失败保留缓存并记录 `.download_failure.json`。本次网络访问返回 403，无法声称已取得最新数据。`data/README.md` 和 `data/inventory.json` 记录本地实际文件及限制；历史镜像只供过期数据的工程回归。

当前机器的 `.venv` 为避免重复下载大型 CUDA 包，复用了已有 GPU Python 包路径，并单独安装交易日历依赖；环境的实际版本见 `environment.json`。在另一台机器应按上述命令安装，不能复制这台机器的绝对路径。

## 运行与预测

```bash
python validate_data.py --csv data/AAPL_1min.csv --profile minute
python audit_archive.py ../../archive.zip
python run_experiment.py --profile annual --csv data/AAPL_1d.csv
python run_experiment.py --profile quarterly --csv data/AAPL_1d.csv
python run_experiment.py --profile monthly --csv data/AAPL_1d.csv --models last_close drift seasonal ridge
python run_experiment.py --profile hourly --csv data/AAPL_1min.csv
python run_experiment.py --profile minute --csv data/AAPL_1min.csv
python audit_user_minute_data.py
python run_experiment.py --profile minute --csv data/AAPL_clean_1min.csv \
  --exploratory-quality-limited --output results/iex_exploratory
python run_experiment.py --profile minute --csv data/AAPL_1min.csv --bar-interval 5min
python predict.py --artifact results/validation/annual/future_close_h12.joblib --csv data/AAPL_1d.csv
python -m pytest tests/test_experiment.py -q
```

默认只运行滚动验证。`--lookbacks`、`--horizons` 和 `--models` 可缩小诊断实验；配置以**交易柱数量**计，不再硬编码 48→12。5 分钟模式默认窗口 6/12/24/48/96，预测 1/3/6/12 柱，即 5/15/30/60 个交易分钟。

结果包括 `REPORT.md`、`report.json`、冻结配置、所选模型的 `.joblib`、验证价格 CSV 和美元价格图。导出文件包含神经网络 state_dict 或 Ridge、特征／目标标准化器、特征顺序、架构、训练截止日期、切分设置和数据／代码哈希。基线无需拟合标准化器。`predict.py` 只接受显式指定的新格式文件，复用训练预处理并输出未来交易柱起始时间、美元收盘价、对数收益率；预测收盘发生在该柱结束时。只加载可信的本地 joblib 文件。旧 `run.py`、`train.py`、`data_processor.py` 保留作历史参考。

当前正式验证结果以 `results/current_validation/README.md` 及其下的三个 profile 报告为准；其中的六个模型 artifact 使用当前代码重新生成，并已通过当前 `predict.py` 的哈希检查。`results/repair_validation/` 保留的是历史验证结果，其 `VERIFICATION.md` 记录旧代码哈希 `44758e...`，不得与当前结果混用。可用 `python summarize_results.py` 重新生成汇总及每折涨跌／波动背景表。价格图每折最多展示 100 个最终预测步样本，指标使用全部有效样本；输入为 log OHLC 和 log1p volume 五个因果通道。

## 时间、来源与缺口

`profiles.json` 定义五种时间尺度及最低历史：年度 15 年、季度 10 年、月度 5 年、小时 1 年、分钟 6 个月。日内默认要求每个完整常规交易日及全期间覆盖率均至少 98%。不足写 `insufficient_data` 并返回 2；无法形成五折的配置记录跳过原因，不生成虚假排名。

用户指定的真实分钟输入是 `data/AAPL_clean_1min.csv`，来源页为 [HF Data Library](https://hfdatalibrary.com/pages/download?)。本地文件哈希绑定的人工来源凭证记录纽约时区、分钟起始标签及提供方所述复权政策；这证明审阅了来源说明，不证明精确下载版本。审计发现总常规时段覆盖率约 97.0%，低于 98% 正式门槛；来源是 IEX 单一交易所，且 Clean 版居中 50 分钟异常值过滤依赖未来邻近柱，无法支持严格因果结论。正式实验仍返回 `insufficient_data`。仅用于开发诊断时可显式添加 `--exploratory-quality-limited`；输出会标记 `exploratory_quality_limited`、不接受任何模型且不运行最终测试。细节见 `data/audits/AAPL_clean_1min_quality.md`。

CSV 至少包含 timestamp/Date 与 Open/High/Low/Close/Volume。排序并保留重复时间的最后一条，记录去重数量；拒绝无效数值和 OHLC 边界。仅用 XNYS 常规时段；日历包含节假日、提前收市、整天缺失和 DST。不填补缺失分钟或闭市时间。输入／标签可跨交易日，闭市跳过，真正缺柱不能压缩。小时柱从开盘起对齐，只保留完整 60 分钟；通常每天 6 柱，末尾 30 分钟不进入小时样本，8 柱预测会跨日。未来时间不是简单加自然小时。由日线生成月线时只保留完整交易月。

日内来源必须有哈希匹配的 sidecar 和凭证。`source_verified: true` 或来源文字本身不构成验证。自动下载凭证由下载入口生成；外部供应商数据需要人工核对原始文件、来源网址、时区、时间标签方向和复权语义，然后创建 `provenance.method=manual_review` 的凭证，记录 reviewer、source_url、review_evidence、timezone、csv_sha256、decision=verified。sidecar 通过 evidence_path / evidence_sha256 引用凭证，并声明 symbol=AAPL、timezone、sha256。模板见 `data/provenance.example.json`。这些是可追溯的本地审计记录，不是对来源真伪的密码学证明；仍需实际检查证据。数据默认应为一分钟、柱起始时间标记；不要把结束标记数据直接声明为起始标记。

用户提供的 `archive.zip` 已全量审计为日线到分钟的线性插值，带未来端点泄漏，不适合真实分钟训练。过滤交易时段也无法修复。审计见 `data/audits/ARCHIVE_AUDIT.md`；原 ZIP 未修改、未上传。

## 验证、选择和最终测试

每个预测跨度使用所有窗口的共同有效 origins，训练／验证样本与目标完全一致，五折扩展训练，每折 purge 等于预测跨度。特征／目标标准化仅拟合训练区间，早停只使用外层训练内部的带 purge 尾段。MASE 分母取共同原始训练历史的一步价格变化，与输入窗口无关。

候选模型须同时满足中位 MASE < 1、相对最后收盘价的中位美元 RMSE 提升至少 2%、至少 3/5 折改善。通过后按归一化对数收益率 RMSE、折间波动、复杂度选择；全部失败就正式选择最后收盘价，其他候选只作探索。报告给出 MAE、RMSE、sMAPE、MASE、方向准确率、每步和每折误差、训练误差及基线提升。诊断与 hybrid 条件详见 [DIAGNOSTICS.md](DIAGNOSTICS.md)。

最终测试默认不运行。验证完成后，在**来源已验证且测试时期未暴露**的数据上显式执行：

```bash
python run_experiment.py --profile annual --csv data/AAPL_1d.csv \
  --final --freeze results/validation/annual/frozen_config.json --output results/final_new
```

配置、数据和代码必须匹配。`results/test_use_ledger.json` 在评估前锁定登记；不同结果目录、源文件哈希和时间尺度都不能使重叠的 AAPL 测试时期恢复未接触。`results/final` 的旧报告是历史已暴露评估，旧选择逻辑有缺陷；请使用修复后的验证结果。不能用已有最终测试结果继续调参。

## 计算资源

CUDA 可用时使用设备 0，在本机验证为 NVIDIA GeForce RTX 5070 Ti Laptop GPU。神经网络批量训练／推断，避免将整份验证张量放到显存。CPU 训练设为 24 threads、24 DataLoader workers，Ridge 的 BLAS 线程限制为 24；CUDA DataLoader 用 0 workers。可用性与版本记录在报告中。
