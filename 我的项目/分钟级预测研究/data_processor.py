"""
数据处理模块：用于数据加载、预处理和特征工程
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from config import Config
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch
import gc

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config: Config):
        """
        初始化数据处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.scaler = StandardScaler()
        self.sequence_length = config.SEQUENCE_LENGTH
        self.prediction_length = config.PREDICTION_LENGTH
        self.feature_columns = config.FEATURE_COLUMNS
        self.target_column = config.TARGET_COLUMN
        self.batch_size = config.BATCH_SIZE
        
    def load_data(self) -> pd.DataFrame:
        """
        加载数据
        
        Returns:
            pd.DataFrame: 加载的数据
        """
        try:
            data_path = Path(self.config.DATA_PATH)
            logger.info(f"正在加载数据: {data_path}")
            df = pd.read_csv(data_path)
            
            # 将第一列设置为时间戳
            df.columns = ['timestamp'] + list(df.columns[1:])
            
            # 确保时间戳列是datetime类型
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            raise
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加时间特征
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 添加时间特征后的数据
        """
        # 添加时间特征
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_market_open'] = ((df.index.hour >= 9) & (df.index.hour < 16)).astype(int)
        
        # 添加周期性时间特征
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
        
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加价格相关特征
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 添加价格特征后的数据
        """
        # 价格变化率
        df['price_change'] = df['close'].pct_change()
        df['price_change_5min'] = df['close'].pct_change(periods=5)
        df['price_change_15min'] = df['close'].pct_change(periods=15)
        
        # 价格波动性
        df['price_volatility'] = df['price_change'].rolling(window=20).std()
        df['high_low_ratio'] = df['high'] / df['low']
        
        # 价格动量
        df['price_momentum'] = df['close'] - df['close'].shift(5)
        df['price_acceleration'] = df['price_change'] - df['price_change'].shift(1)
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加成交量相关特征
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 添加成交量特征后的数据
        """
        # 成交量变化
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # 成交量趋势
        df['volume_trend'] = df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=20).mean()
        
        # 价格-成交量关系
        df['price_volume_ratio'] = df['close'] * df['volume']
        df['price_volume_trend'] = df['price_volume_ratio'].pct_change()
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 添加技术指标后的数据
        """
        # 确保数据按时间排序
        df = df.sort_index()
        
        # 计算移动平均线
        sma = SMAIndicator(close=df['close'], window=self.config.SMA_WINDOW)
        ema = EMAIndicator(close=df['close'], window=self.config.EMA_WINDOW)
        df['sma'] = sma.sma_indicator()
        df['ema'] = ema.ema_indicator()
        
        # 计算MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # 计算RSI
        rsi = RSIIndicator(close=df['close'], window=self.config.RSI_WINDOW)
        df['rsi'] = rsi.rsi()
        
        # 计算布林带
        bb = BollingerBands(close=df['close'], window=self.config.BB_WINDOW)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        
        # 计算随机指标
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # 计算VWAP
        vwap = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
        df['vwap'] = vwap.volume_weighted_average_price()
        
        # 计算ATR
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr'] = atr.average_true_range()
        
        # 计算OBV
        obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
        df['obv'] = obv.on_balance_volume()
        
        # 处理缺失值
        df = df.ffill().bfill()  # 使用前向填充和后向填充
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备特征和标签
        
        Args:
            df: 原始数据
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 特征和标签
        """
        # 添加所有特征
        df = self.add_time_features(df)
        df = self.add_price_features(df)
        df = self.add_volume_features(df)
        df = self.calculate_technical_indicators(df)
        
        # 选择特征
        feature_columns = [
            # 原始价格和成交量
            'open', 'high', 'low', 'close', 'volume',
            
            # 时间特征
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_market_open',
            
            # 价格特征
            'price_change', 'price_change_5min', 'price_change_15min',
            'price_volatility', 'high_low_ratio', 'price_momentum', 'price_acceleration',
            
            # 成交量特征
            'volume_change', 'volume_ma', 'volume_ratio',
            'volume_trend', 'price_volume_ratio', 'price_volume_trend',
            
            # 技术指标
            'sma', 'ema', 'macd', 'macd_signal', 'macd_diff',
            'rsi', 'bb_high', 'bb_low', 'bb_mid',
            'stoch_k', 'stoch_d', 'vwap', 'atr', 'obv'
        ]
        
        # 标准化特征
        features = self.scaler.fit_transform(df[feature_columns])
        
        # 准备标签（未来PREDICTION_LENGTH步的所有特征）
        labels = []
        for i in range(len(features) - self.prediction_length):
            labels.append(features[i+1:i+1+self.prediction_length])  # shape: (5, 37)
        labels = np.array(labels)  # shape: (样本数, 5, 37)
        features = features[:len(labels)]
        return features, labels
    
    def create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建序列数据，使用分批处理减少内存使用
        
        Args:
            features: 特征数据
            labels: 标签数据
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 序列特征和标签
        """
        logger.info("开始创建序列数据...")
        
        # 计算每个批次的大小
        total_samples = len(features) - self.sequence_length - self.prediction_length + 1
        samples_per_batch = min(10000, total_samples)  # 每批最多10000个样本
        num_batches = (total_samples + samples_per_batch - 1) // samples_per_batch
        
        X_batches = []
        y_batches = []
        
        for i in range(num_batches):
            start_idx = i * samples_per_batch
            end_idx = min((i + 1) * samples_per_batch, total_samples)
            
            # 创建当前批次的序列
            X_batch = []
            y_batch = []
            
            for j in range(start_idx, end_idx):
                X_batch.append(features[j:j + self.sequence_length])
                y_batch.append(labels[j + self.sequence_length])  # 只取一个[5, 37]标签
            
            X_batches.append(np.array(X_batch, dtype=np.float32))  # 使用float32减少内存使用
            y_batches.append(np.array(y_batch, dtype=np.float32))
            
            # 清理内存
            gc.collect()
            
            logger.info(f"已处理 {end_idx}/{total_samples} 个样本")
        
        # 合并所有批次
        X = np.concatenate(X_batches, axis=0)
        y = np.concatenate(y_batches, axis=0)
        
        # 清理内存
        del X_batches, y_batches
        gc.collect()
        
        logger.info(f"序列数据创建完成，形状: X={X.shape}, y={y.shape}")
        return X, y
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备训练和验证数据
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            (训练集特征, 训练集标签, 验证集特征, 验证集标签)
        """
        # 加载数据
        df = self.load_data()
        # 提取全部特征和标签
        features, labels = self.prepare_features(df)
        # 创建序列
        X, y = self.create_sequences(features, labels)
        # 划分训练集和验证集
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        logger.info(f"数据准备完成: X_train={X_train.shape}, y_train={y_train.shape}")
        return X_train, y_train, X_val, y_val

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        初始化时间序列数据集
        
        Args:
            X: 特征数据
            y: 标签数据
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx] 