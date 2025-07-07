"""
注意力融合模块：用于多尺度特征融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class MultiScaleAttentionFusion(nn.Module):
    def __init__(self, feature_dims: List[int], hidden_dim: int, num_heads: int = 8):
        """
        多尺度注意力融合模块
        
        Args:
            feature_dims: 各尺度特征的维度列表
            hidden_dim: 融合后的隐藏维度
            num_heads: 注意力头数
        """
        super(MultiScaleAttentionFusion, self).__init__()
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 特征投影层
        self.feature_projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in feature_dims
        ])
        
        # 多头注意力层
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 融合后的输出层
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 多尺度特征列表，每个元素形状为 [batch_size, seq_len, feature_dim]
            
        Returns:
            torch.Tensor: 融合后的特征 [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len = features[0].shape[:2]
        
        # 1. 特征投影到统一维度
        projected_features = []
        for i, (feature, projection) in enumerate(zip(features, self.feature_projections)):
            projected = projection(feature)  # [batch_size, seq_len, hidden_dim]
            projected_features.append(projected)
        
        # 2. 拼接多尺度特征
        concatenated = torch.cat(projected_features, dim=1)  # [batch_size, seq_len*num_scales, hidden_dim]
        
        # 3. 多头自注意力
        attended, attention_weights = self.multihead_attention(
            concatenated, concatenated, concatenated
        )
        
        # 4. 残差连接和层归一化
        attended = self.layer_norm1(concatenated + attended)
        
        # 5. 输出投影
        output = self.output_projection(attended)
        output = self.layer_norm2(attended + output)
        
        # 6. 重塑回原始序列长度（取平均值）
        output = output.view(batch_size, len(self.feature_dims), seq_len, self.hidden_dim)
        output = output.mean(dim=1)  # [batch_size, seq_len, hidden_dim]
        
        return output

class GatedAttentionFusion(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        门控注意力融合模块
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏维度
        """
        super(GatedAttentionFusion, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 注意力权重计算
        self.attention_weights = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: 融合后的特征 [batch_size, seq_len, hidden_dim]
        """
        # 计算注意力权重
        attention_weights = self.attention_weights(x)  # [batch_size, seq_len, 1]
        
        # 加权特征
        weighted_features = x * attention_weights  # [batch_size, seq_len, input_dim]
        
        # 门控机制
        gate = self.gate(x)  # [batch_size, seq_len, hidden_dim]
        
        # 特征变换
        transformed = self.output_projection(weighted_features)  # [batch_size, seq_len, hidden_dim]
        
        # 门控融合
        output = gate * transformed
        
        return output

class CrossScaleAttention(nn.Module):
    def __init__(self, scale_dims: List[int], hidden_dim: int):
        """
        跨尺度注意力模块
        
        Args:
            scale_dims: 各尺度的特征维度
            hidden_dim: 统一隐藏维度
        """
        super(CrossScaleAttention, self).__init__()
        self.scale_dims = scale_dims
        self.hidden_dim = hidden_dim
        self.num_scales = len(scale_dims)
        
        # 各尺度的投影层
        self.scale_projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in scale_dims
        ])
        
        # 跨尺度注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            scale_features: 各尺度特征列表
            
        Returns:
            torch.Tensor: 融合后的特征
        """
        # 投影到统一维度
        projected_features = []
        for feature, projection in zip(scale_features, self.scale_projections):
            projected = projection(feature)
            projected_features.append(projected)
        
        # 拼接所有尺度特征
        concatenated = torch.cat(projected_features, dim=1)  # [batch, seq_len*num_scales, hidden_dim]
        
        # 跨尺度注意力
        attended, _ = self.cross_attention(concatenated, concatenated, concatenated)
        
        # 输出投影
        output = self.output_layer(attended)
        
        return output 