"""
EMA分解模块：用于将时间序列分解为趋势和残差成分
"""

import torch
import torch.nn as nn

class EMADecomposition(nn.Module):
    def __init__(self, input_dim: int, alpha: float = 0.1):
        """
        初始化EMA分解模块
        
        Args:
            input_dim: 输入特征维度
            alpha: EMA平滑系数
        """
        super().__init__()
        self.alpha = alpha
        self.input_dim = input_dim
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: 趋势和残差成分
        """
        # 计算EMA
        ema = torch.zeros_like(x)
        ema[:, 0] = x[:, 0]
        
        for t in range(1, x.size(1)):
            ema[:, t] = self.alpha * x[:, t] + (1 - self.alpha) * ema[:, t-1]
        
        # 趋势成分
        trend = ema
        
        # 残差成分
        residual = x - trend
        
        return trend, residual 