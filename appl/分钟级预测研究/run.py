import os
import sys
import subprocess

HELP = """
用法: python run.py [--quick|--full|--compare|--predict|--all]

--quick    仅快速训练验证
--full     完整训练
--compare  模型对比实验
--predict  预测与回测
--all      全流程（默认）
"""


def run_cmd(cmd):
    print(f"\n>>> 正在执行: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"命令失败: {cmd}")
        sys.exit(1)

def main():
    args = sys.argv[1:]
    mode = '--all' if not args else args[0]
    if mode not in ['--quick', '--full', '--compare', '--predict', '--all', '-h', '--help']:
        print(HELP)
        return
    if mode in ['-h', '--help']:
        print(HELP)
        return

    print("\n==== 高频交易预测系统一键运行 ====")
    print("当前模式:", mode)

    # 步骤1: 检查依赖
    print("\n[1/6] 检查依赖...")
    # 自动定位 requirements.txt 路径
    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')
    if not os.path.exists(req_path):
        print(f"未找到 requirements.txt，已跳过依赖安装。请手动检查依赖。\n实际查找路径: {req_path}")
    else:
        run_cmd(f'pip install -r "{req_path}"')

    # 步骤2: 数据检查与特征工程
    print("\n[2/6] 数据检查与特征工程...")
    # data_processor.py 通常被train.py等自动调用，无需单独执行

    # 步骤3: 快速训练
    if mode in ['--quick', '--all']:
        print("\n[3/6] 快速训练验证...")
        run_cmd('python quick_train.py')

    # 步骤4: 完整训练
    if mode in ['--full', '--all']:
        print("\n[4/6] 完整训练...")
        run_cmd('python train.py')

    # 步骤5: 模型对比
    if mode in ['--compare', '--all']:
        print("\n[5/6] 模型对比实验...")
        run_cmd('python compare_models_enhanced.py')

    # 步骤6: 预测与回测
    if mode in ['--predict', '--all']:
        print("\n[6/6] 预测与回测...")
        run_cmd('python predict.py')

    print("\n==== 全部流程执行完毕！====\n")
    print("结果请查看 plots/、checkpoints/、quick_train_report.txt 等输出文件。\n")

if __name__ == '__main__':
    main() 