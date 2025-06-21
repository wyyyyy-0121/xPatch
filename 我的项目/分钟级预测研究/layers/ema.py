"""
EMA分解层：实现指数移动平均分解
"""

import torch
import torch.nn as nn

class EMADecomposition(nn.Module):
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        使用EMA进行时间序列分解
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            
        Returns:
            tuple: (trend, seasonality)
                - trend: 趋势成分 [batch_size, seq_len, input_dim]
                - seasonality: 季节性成分 [batch_size, seq_len, input_dim]
        """
        # 计算趋势（使用EMA）
        trend = torch.zeros_like(x)
        trend[:, 0] = x[:, 0]
        for t in range(1, x.size(1)):
            trend[:, t] = self.alpha * x[:, t] + (1 - self.alpha) * trend[:, t-1]
        
        # 季节性成分 = 原始序列 - 趋势
        seasonality = x - trend
        
        return trend, seasonality 