"""
一键运行脚本：自动处理所有步骤
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime

# 获取当前脚本所在目录的绝对路径
CURRENT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = CURRENT_DIR.parent

# 配置日志
def setup_logging():
    """设置日志"""
    log_dir = CURRENT_DIR / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_environment():
    """检查环境"""
    logger.info("检查Python环境...")
    try:
        import numpy
        import pandas
        import torch
        import sklearn
        import matplotlib
        import seaborn
        import ta
        logger.info("基础依赖检查通过")
        return True
    except ImportError as e:
        logger.error(f"缺少必要的依赖包: {str(e)}")
        logger.info("正在安装依赖...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(CURRENT_DIR / "requirements.txt")])
            logger.info("依赖安装完成")
            return True
        except Exception as e:
            logger.error(f"依赖安装失败: {str(e)}")
            return False

def create_directories():
    """创建必要的目录"""
    logger.info("创建必要的目录...")
    directories = ['data', 'models', 'layers', 'checkpoints', 'plots', 'logs']
    for dir_name in directories:
        dir_path = CURRENT_DIR / dir_name
        dir_path.mkdir(exist_ok=True)
        logger.info(f"确保目录存在: {dir_path}")

def check_data():
    """检查数据文件"""
    logger.info("检查数据文件...")
    
    # 尝试多个可能的数据文件位置
    possible_paths = [
        CURRENT_DIR / 'data' / 'AAPL_1min.csv',  # 当前目录下的data文件夹
        PROJECT_ROOT / 'data' / 'AAPL_1min.csv',  # 项目根目录下的data文件夹
        Path('data/AAPL_1min.csv'),  # 相对路径
        Path('分钟级预测研究/data/AAPL_1min.csv')  # 带项目名的相对路径
    ]
    
    data_file = None
    for path in possible_paths:
        if path.exists():
            data_file = path
            logger.info(f"找到数据文件: {path}")
            break
    
    if data_file is None:
        logger.error("未找到数据文件，已尝试以下路径:")
        for path in possible_paths:
            logger.error(f"- {path}")
        return False
    
    # 检查数据文件格式
    try:
        import pandas as pd
        df = pd.read_csv(data_file)
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"数据文件格式错误，需要包含以下列: {required_columns}")
            return False
        
        # 检查数据量是否足够
        if len(df) < 1000:
            logger.warning(f"数据量较少，当前只有 {len(df)} 条记录")
        
        logger.info(f"数据文件检查通过，共 {len(df)} 条记录")
        return True
    except Exception as e:
        logger.error(f"数据文件检查失败: {str(e)}")
        return False

def run_script(script_name):
    """运行Python脚本"""
    logger.info(f"\n运行 {script_name}...")
    script_path = CURRENT_DIR / script_name
    try:
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=True, 
                              text=True)
        if result.returncode == 0:
            logger.info(f"{script_name} 运行成功")
            if result.stdout:
                logger.info("输出信息:")
                logger.info(result.stdout)
            return True
        else:
            logger.error(f"{script_name} 运行失败:")
            logger.error(result.stderr)
            return False
    except Exception as e:
        logger.error(f"运行 {script_name} 时出错: {str(e)}")
        return False

def main():
    """主函数"""
    logger.info("=== 开始运行分钟级预测系统 ===")
    
    # 1. 检查环境
    if not check_environment():
        logger.error("环境检查失败，请手动安装依赖")
        return
    
    # 2. 创建目录
    create_directories()
    
    # 3. 检查数据
    if not check_data():
        logger.error("数据文件检查失败，请确保数据文件存在且格式正确")
        return
    
    # 4. 运行数据处理
    if not run_script('data_processor.py'):
        logger.error("数据处理失败，程序终止")
        return
    
    # 5. 运行训练
    if not run_script('train.py'):
        logger.error("模型训练失败，程序终止")
        return
    
    # 6. 运行预测
    if not run_script('predict.py'):
        logger.error("预测失败，程序终止")
        return
    
    logger.info("\n=== 所有步骤已完成 ===")
    logger.info("结果文件保存在以下位置：")
    logger.info("- 预测图表: plots/predictions.png")
    logger.info("- 预测vs实际值对比图: plots/prediction_vs_actual.png")
    logger.info("- 损失曲线: plots/loss_curve.png")
    logger.info(f"- 运行日志: {log_file}")

if __name__ == "__main__":
    # 设置日志
    logger = setup_logging()
    log_file = CURRENT_DIR / 'logs' / f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    try:
        main()
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}", exc_info=True)
    finally:
        logger.info("程序运行结束")
        input("按回车键退出...") 