import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_data(df):
    """验证数据格式和内容"""
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # 检查必要的列是否存在
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"数据缺少必要的列: {missing_cols}")
    
    # 检查数据类型
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 检查缺失值
    missing_values = df.isnull().sum()
    if missing_values.any():
        logger.warning("数据中存在缺失值:")
        logger.warning(missing_values[missing_values > 0])
    
    # 检查数据范围
    for col in numeric_cols:
        if (df[col] < 0).any():
            logger.warning(f"列 {col} 中存在负值")
    
    return df

def main():
    """主函数"""
    try:
        # 读取数据
        input_file = Path("data/AAPL_1min.csv")
        logger.info(f"正在读取数据文件: {input_file}")
        df = pd.read_csv(input_file)
        
        # 验证数据
        logger.info("正在验证数据...")
        df = validate_data(df)
        
        logger.info(f"数据验证完成！共 {len(df)} 条记录")
        return True
        
    except Exception as e:
        logger.error(f"处理数据时发生错误: {str(e)}")
        return False

if __name__ == "__main__":
    main() 