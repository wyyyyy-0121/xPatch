"""
残差流模块：处理时间序列的残差成分
"""

import torch
import torch.nn as nn

class ResidualStream(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, d_ff: int, dropout: float):
        """
        初始化残差流模块
        
        Args:
            hidden_size: 隐藏层维度
            num_layers: Transformer层数
            num_heads: 注意力头数
            d_ff: 前馈网络维度
            dropout: Dropout比率
        """
        super().__init__()
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_size]
            
        Returns:
            torch.Tensor: 处理后的张量 [batch_size, seq_len, hidden_size]
        """
        # 创建注意力掩码（允许所有位置相互关注）
        mask = None
        
        # 通过Transformer编码器
        output = self.transformer(x, mask=mask)
        
        return output 