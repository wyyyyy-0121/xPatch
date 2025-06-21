"""
xPatch模型实现
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from .ema_decomposition import EMADecomposition
from .patch_embedding import PatchEmbedding
from .trend_stream import TrendStream
from .residual_stream import ResidualStream

class XPatch(nn.Module):
    def __init__(self, input_dim: int, config: Any):
        """
        初始化xPatch模型
        
        Args:
            input_dim: 输入特征维度
            config: 配置对象
        """
        super().__init__()
        
        # EMA分解
        self.ema_decomposition = EMADecomposition(
            input_dim=input_dim,
            alpha=config.EMA_ALPHA
        )
        
        # Patch嵌入
        self.patch_embedding = PatchEmbedding(
            input_dim=input_dim,
            hidden_size=config.HIDDEN_SIZE,
            patch_sizes=config.PATCH_SIZES
        )
        
        # 趋势流
        self.trend_stream = TrendStream(
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            num_heads=config.NUM_HEADS,
            d_ff=config.D_FF,
            dropout=config.DROPOUT
        )
        
        # 残差流
        self.residual_stream = ResidualStream(
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            num_heads=config.NUM_HEADS,
            d_ff=config.D_FF,
            dropout=config.DROPOUT
        )
        
        # 回归头
        self.regression_head = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_SIZE // 2, config.PREDICTION_LENGTH)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            
        Returns:
            Dict[str, torch.Tensor]: 包含预测结果的字典
        """
        # EMA分解
        trend, residual = self.ema_decomposition(x)
        
        # Patch嵌入
        trend_embedded = self.patch_embedding(trend)
        residual_embedded = self.patch_embedding(residual)
        
        # 趋势流处理
        trend_output = self.trend_stream(trend_embedded)
        
        # 残差流处理
        residual_output = self.residual_stream(residual_embedded)
        
        # 合并输出
        combined = trend_output + residual_output
        
        # 回归预测
        regression_pred = self.regression_head(combined[:, -1])
        
        return {
            'regression_pred': regression_pred,
            'trend': trend,
            'residual': residual
        }
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测函数
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 预测结果
        """
        outputs = self.forward(x)
        return outputs['regression_pred'] 