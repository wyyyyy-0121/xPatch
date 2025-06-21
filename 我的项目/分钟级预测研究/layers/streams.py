"""
双流网络：实现线性流和非线性流
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearStream(nn.Module):
    """线性流：使用MLP处理趋势成分"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # 隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 趋势成分 [batch_size, seq_len, input_dim]
        Returns:
            torch.Tensor: 趋势特征 [batch_size, hidden_dim]
        """
        # 对序列维度取平均
        x = x.mean(dim=1)  # [batch_size, input_dim]
        return self.mlp(x)  # [batch_size, hidden_dim]

class NonLinearStream(nn.Module):
    """非线性流：使用CNN处理季节性成分"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        
        # CNN层
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim if i == 0 else hidden_dim, 
                         hidden_dim, 
                         kernel_size=3, 
                         padding=1),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 季节性成分 [batch_size, seq_len, input_dim]
        Returns:
            torch.Tensor: 季节性特征 [batch_size, hidden_dim]
        """
        # 转置以适应CNN
        x = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]
        
        # CNN处理
        for conv in self.conv_layers:
            x = conv(x)
            
        # 全局平均池化
        x = x.mean(dim=2)  # [batch_size, hidden_dim]
        return x 