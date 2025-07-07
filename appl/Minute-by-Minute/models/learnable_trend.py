"""
可学习的趋势提取模块：替代固定的EMA分解
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class LearnableTrendExtraction(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        """
        可学习的趋势提取模块
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏维度
            num_layers: 层数
        """
        super(LearnableTrendExtraction, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 趋势提取网络
        self.trend_encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True
        )
        
        # 趋势预测头
        self.trend_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 门控机制
        self.trend_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (趋势成分, 残差成分)
        """
        # 双向LSTM编码
        trend_encoded, _ = self.trend_encoder(x)  # [batch_size, seq_len, hidden_dim*2]
        
        # 预测趋势
        trend_pred = self.trend_head(trend_encoded)  # [batch_size, seq_len, input_dim]
        
        # 门控机制
        gate = self.trend_gate(x)  # [batch_size, seq_len, hidden_dim]
        gate = gate.unsqueeze(-1).expand(-1, -1, self.input_dim)  # [batch_size, seq_len, input_dim]
        
        # 应用门控
        trend = trend_pred * gate
        residual = x - trend
        
        return trend, residual

class GatedConvTrendExtraction(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_sizes: list = [3, 5, 7]):
        """
        门控卷积趋势提取模块
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏维度
            kernel_sizes: 卷积核大小列表
        """
        super(GatedConvTrendExtraction, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_sizes = kernel_sizes
        
        # 多尺度卷积层
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        # 门控卷积层
        self.gate_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        
        # 趋势预测头
        self.trend_head = nn.Sequential(
            nn.Linear(hidden_dim * len(kernel_sizes), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (趋势成分, 残差成分)
        """
        # 转置为卷积格式
        x_conv = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]
        
        # 多尺度卷积特征
        conv_features = []
        for conv in self.conv_layers:
            conv_feat = conv(x_conv)  # [batch_size, hidden_dim, seq_len]
            conv_features.append(conv_feat)
        
        # 门控卷积
        gate = torch.sigmoid(self.gate_conv(x_conv))  # [batch_size, hidden_dim, seq_len]
        
        # 应用门控
        gated_features = []
        for conv_feat in conv_features:
            gated_feat = conv_feat * gate
            gated_features.append(gated_feat)
        
        # 拼接多尺度特征
        concatenated = torch.cat(gated_features, dim=1)  # [batch_size, hidden_dim*num_kernels, seq_len]
        concatenated = concatenated.transpose(1, 2)  # [batch_size, seq_len, hidden_dim*num_kernels]
        
        # 预测趋势
        trend = self.trend_head(concatenated)  # [batch_size, seq_len, input_dim]
        
        # 计算残差
        residual = x - trend
        
        return trend, residual

class AdaptiveTrendExtraction(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        自适应趋势提取模块
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏维度
        """
        super(AdaptiveTrendExtraction, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 短期趋势提取
        self.short_trend = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)
        
        # 长期趋势提取
        self.long_trend = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True)
        
        # 自适应权重
        self.adaptive_weight = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 趋势融合
        self.trend_fusion = nn.Linear(hidden_dim * 2, input_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (趋势成分, 残差成分)
        """
        # 短期趋势
        short_trend, _ = self.short_trend(x)  # [batch_size, seq_len, hidden_dim]
        
        # 长期趋势
        long_trend, _ = self.long_trend(x)  # [batch_size, seq_len, hidden_dim]
        
        # 自适应权重
        combined = torch.cat([short_trend, long_trend], dim=-1)  # [batch_size, seq_len, hidden_dim*2]
        weight = self.adaptive_weight(combined)  # [batch_size, seq_len, 1]
        
        # 加权融合
        weighted_short = short_trend * weight
        weighted_long = long_trend * (1 - weight)
        
        # 趋势融合
        trend_features = torch.cat([weighted_short, weighted_long], dim=-1)
        trend = self.trend_fusion(trend_features)  # [batch_size, seq_len, input_dim]
        
        # 计算残差
        residual = x - trend
        
        return trend, residual 