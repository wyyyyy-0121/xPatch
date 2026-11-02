# AAPL数据取得工具

2026-11-02 的实际 SIP 访问与 20 日验收已完成，见 [v2 供应商验收](V2_SOURCE_AUDIT.md) 和仓库根 [执行台账](../EXECUTION_PLAN.md)。后续使用 `~/.venvs/xpatch-v2/bin/python`，从仓库根以模块形式运行审计器。

下载器会在环境变量不存在时读取本地 `~/.config/alpaca/credentials.env` 的两个字面赋值；不执行 shell 文件。支持 `--credentials-file` 指定文件，以及 `--resume --output <原目录>` 校验已取得的原始分页后续传。最多 4 个日期范围并发，各范围仍按分页顺序请求。

```bash
~/.venvs/xpatch-v2/bin/python -m data_acquisition.audit_alpaca_sip \
  --run data_acquisition/runs/alpaca_audit20_20261102T0800Z
~/.venvs/xpatch-v2/bin/python -m data_acquisition.audit_history \
  --run data_acquisition/runs/alpaca_development_20261102
```

`audit_history` 要求下载完整；会从原始响应重建 CSV 并核对全部 OHLCV、trade_count、vwap 字段，然后按请求完整日期范围审计（包含整日漏数）。覆盖合格不自动意味着正式点时研究资格。

当前成果和限制见[获取报告](REPORT.md)。公开FirstRate Data短样本已下载并做结构审计；多年历史SIP尚需账户权限实测。

## 使用已有环境

在WSL Ubuntu终端：

~~~bash
cd /home/ayaka/xPatch
appl/Min-by-Min/.venv/bin/python data_acquisition/download_alpaca_sip.py --dry-run
~~~

默认列出20个审计交易日，不发网络请求。下载器本身只用Python标准库；样本审计需要已有环境中的pandas、numpy、exchange_calendars。

## 配置账户后取得20日SIP样本

无需把密钥发进聊天，也不要写进命令行或Git文件。已有Alpaca账户时，在自己的WSL本地终端运行：

~~~bash
cd /home/ayaka/xPatch
appl/Min-by-Min/.venv/bin/python data_acquisition/download_alpaca_sip.py --prompt-credentials
~~~

两个输入均隐藏，不持久保存。也支持已配置的APCA_API_KEY_ID与APCA_API_SECRET_KEY环境变量。脚本只访问行情接口，不注册、购买、下单或改变订阅；认证或SIP权限失败立即报错，不回退IEX。历史SIP权限必须对实际账户验证。

原始响应和CSV默认进入runs/alpaca_<UTC时间>，全部被Git忽略。失败保留receipt.json说明完整性，不能将未完成下载当合格数据。采集成功仍需覆盖、异常和来源审核，download_complete不等于source_verified。

完成20日验收后，多年开发历史候选命令：

~~~bash
appl/Min-by-Min/.venv/bin/python data_acquisition/download_alpaca_sip.py \
  --prompt-credentials --start 2021-01-04 --end 2025-03-25
~~~

脚本按不超过30自然日分段、逐页下载，显式请求SIP与raw adjustment，保存全部原始响应。此命令不作为首次样本请求，也不代表多年下载已经执行。日期范围均为开发用途，不创建独立测试集。

## 已取得的公开样本

原始文件及结构审计位于runs/20261029T133057666130Z，见[报告](REPORT.md)。原始行情禁止未经许可再分发，不上传CSV/ZIP到GitHub。两个公开链接来自同一供应商，不是第二来源验证。

probe_public_sample.py可取得产品页样本与官网说明，每次保存新版本及哈希；公开链接内容可能更新。audit_public_sample.py用于已解出、列名为timestamp/open/high/low/close/volume的FirstRate Data评价样本，不能泛化为所有供应商审计器。
