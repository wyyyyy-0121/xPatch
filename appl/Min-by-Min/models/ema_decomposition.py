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
        if not 0 <= alpha <= 1:
            raise ValueError('EMA alpha must be in [0, 1]')
        self.register_buffer('_weights', torch.empty(0), persistent=False)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: 趋势和残差成分
        """
        # 计算EMA
        length = x.size(1)
        if (self._weights.shape != (length, length) or self._weights.device != x.device
                or self._weights.dtype != x.dtype):
            t = torch.arange(length, device=x.device)
            distance = t[:, None] - t[None, :]
            weights = self.alpha * (1-self.alpha) ** distance.clamp_min(0).to(x.dtype)
            weights = weights * (distance >= 0)
            weights[:, 0] = (1-self.alpha) ** t.to(x.dtype)
            self._weights = weights
        # Closed-form causal recurrence; avoids one GPU kernel launch per step.
        ema = torch.matmul(self._weights, x)
        
        # 趋势成分
        trend = ema
        
        # 残差成分
        residual = x - trend
        
        return trend, residual 
