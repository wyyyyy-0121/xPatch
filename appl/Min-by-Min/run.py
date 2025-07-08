import os
import sys
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()

def run_cmd(cmd):
    print(f"\n>>> Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=PROJECT_DIR)
    if result.returncode != 0:
        print(f"[错误] 命令执行失败: {cmd}")
        sys.exit(1)

def parse_compare_results(report_path):
    best_model = None
    best_mae = float('inf')
    summary = ""
    if not report_path.exists():
        print("[警告] 未找到模型对比报告，无法自动总结最佳模型。")
        return
    with open(report_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "MAE=" in line:
                parts = line.strip().split()
                model = parts[0].replace(":", "")
                try:
                    mae = float(parts[1].split("=")[1])
                except Exception:
                    continue
                if mae < best_mae:
                    best_mae = mae
                    best_model = model
    if best_model:
        summary = f"本次对比中，最佳模型为【{best_model}】，其MAE最低（{best_mae:.6f}），预测效果最优。\n"
        if best_model.lower() == "xpatch":
            summary += "优点：xPatch结构融合了多尺度特征和趋势残差双流，适合高频时序数据，泛化能力强。"
        elif best_model.lower() == "lstm":
            summary += "优点：LSTM善于捕捉时序依赖，适合处理长序列数据。"
        elif best_model.lower() == "gru":
            summary += "优点：GRU结构简洁，训练速度快，适合资源有限场景。"
        elif best_model.lower() == "mlp":
            summary += "优点：MLP结构简单，适合特征工程充分的场景。"
        else:
            summary += "优点：该模型在本数据集上表现突出。"
    print("\n==== 模型对比自动总结 ====")
    print(summary)

def main():
    os.chdir(PROJECT_DIR)
    print("\n==== xPatch 高频交易预测系统一键主流程 ====")
    print("工作目录:", os.getcwd())

    # 1. 安装依赖
    run_cmd('pip install -r requirements.txt')

    # 2. 完整训练
    run_cmd('python train.py')

    # 3. 多模型对比
    run_cmd('python compare_models_enhanced.py')

    # 4. 预测与回测
    run_cmd('python predict.py')

    # 5. 可视化（如有）
    if (PROJECT_DIR / 'visualization.py').exists():
        run_cmd('python visualization.py')

    # 6. 自动输出最佳模型及优点
    parse_compare_results(PROJECT_DIR / 'model_comparison_report.txt')

    print("\n==== 全部流程已完成！ ====")
    print("请查看 plots/、checkpoints/、model_comparison_report.txt、logs/ 等目录和文件获取结果。\n")

if __name__ == '__main__':
    main() 