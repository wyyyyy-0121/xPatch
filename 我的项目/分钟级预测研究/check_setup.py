"""
检查项目设置是否正确
"""

import os
import sys
import pandas as pd
from pathlib import Path

def check_environment():
    """检查Python环境和必要的包"""
    print("检查Python环境...")
    try:
        import torch
        import numpy
        import pandas
        import matplotlib
        print("✓ 所有必要的包都已安装")
        return True
    except ImportError as e:
        print(f"✗ 缺少必要的包: {str(e)}")
        return False

def check_data_files():
    """检查数据文件"""
    print("\n检查数据文件...")
    data_dir = Path("data")
    required_files = ["AAPL_1min.csv"]
    
    if not data_dir.exists():
        print(f"✗ 数据目录不存在: {data_dir}")
        return False
    
    all_files_exist = True
    for file in required_files:
        file_path = data_dir / file
        if not file_path.exists():
            print(f"✗ 数据文件不存在: {file_path}")
            all_files_exist = False
        else:
            print(f"✓ 找到数据文件: {file_path}")
            # 检查文件格式
            try:
                df = pd.read_csv(file_path)
                required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_columns):
                    print(f"✗ 数据文件格式不正确，缺少必要的列: {required_columns}")
                    all_files_exist = False
                else:
                    print(f"✓ 数据文件格式正确，包含 {len(df)} 条记录")
            except Exception as e:
                print(f"✗ 无法读取数据文件: {str(e)}")
                all_files_exist = False
    
    return all_files_exist

def check_model_files():
    """检查模型文件"""
    print("\n检查模型文件...")
    model_dir = Path("models")
    if not model_dir.exists():
        print(f"✗ 模型目录不存在: {model_dir}")
        return False
    
    print("✓ 模型目录存在")
    return True

def check_checkpoints():
    """检查检查点文件"""
    print("\n检查检查点文件...")
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        print(f"✗ 检查点目录不存在: {checkpoint_dir}")
        return False
    
    print("✓ 检查点目录存在")
    return True

def main():
    """主函数"""
    print("开始检查项目设置...\n")
    
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent
    os.chdir(current_dir)
    
    checks = [
        ("环境检查", check_environment),
        ("数据文件检查", check_data_files),
        ("模型文件检查", check_model_files),
        ("检查点检查", check_checkpoints)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n=== {check_name} ===")
        if not check_func():
            all_passed = False
    
    print("\n=== 检查结果汇总 ===")
    if all_passed:
        print("✓ 所有检查都通过了！项目可以正常运行。")
    else:
        print("✗ 部分检查未通过，请解决上述问题后再运行项目。")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 