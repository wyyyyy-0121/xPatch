"""
Patch嵌入模块：将时间序列数据转换为patch序列
"""

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, patch_sizes: list[int]):
        """
        初始化Patch嵌入模块
        
        Args:
            input_dim: 输入特征维度
            hidden_size: 隐藏层维度
            patch_sizes: patch大小列表
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.patch_sizes = patch_sizes
        
        # 为每个patch大小创建投影层
        self.projections = nn.ModuleList([
            nn.Linear(patch_size * input_dim, hidden_size)
            for patch_size in patch_sizes
        ])
        
        # 位置编码
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1000, hidden_size)  # 假设最大序列长度为1000
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: 嵌入后的张量 [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.size()
        patch_embeddings = []
        patch_lengths = []
        # 对每个patch大小进行处理
        for i, patch_size in enumerate(self.patch_sizes):
            patches = []
            for j in range(0, seq_len - patch_size + 1):
                patch = x[:, j:j+patch_size, :].reshape(batch_size, -1)
                patches.append(patch)
            patches = torch.stack(patches, dim=1)  # [batch, patch_num, patch_size*input_dim]
            patch_embedding = self.projections[i](patches)  # [batch, patch_num, hidden_size]
            patch_embeddings.append(patch_embedding)
            patch_lengths.append(patch_embedding.shape[1])
        # 对齐所有patch序列长度
        min_len = min(patch_lengths)
        patch_embeddings = [pe[:, :min_len, :] for pe in patch_embeddings]
        combined = torch.stack(patch_embeddings, dim=1).mean(dim=1)  # [batch, min_len, hidden_size]
        # 添加位置编码
        combined = combined + self.pos_embedding[:, :combined.size(1), :]
        return combined 