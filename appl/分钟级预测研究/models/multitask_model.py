"""
多任务学习模型：同时进行价格回归预测和涨跌方向分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class MultiTaskLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, prediction_length, dropout=0.2):
        super(MultiTaskLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_length = prediction_length
        
        # 共享LSTM编码器
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
        # 回归头：预测价格
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, prediction_length)
        )
        
        # 分类头：预测涨跌方向
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, prediction_length * 3)  # 3类：涨、跌、平
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        encoded = self.dropout(last_output)
        
        # 回归预测
        regression_pred = self.regression_head(encoded)
        
        # 分类预测
        classification_logits = self.classification_head(encoded)
        classification_logits = classification_logits.view(-1, self.prediction_length, 3)
        classification_probs = F.softmax(classification_logits, dim=-1)
        
        return {
            'regression_pred': regression_pred,
            'classification_logits': classification_logits,
            'classification_probs': classification_probs
        }
    
    def predict(self, x):
        """预测函数"""
        outputs = self.forward(x)
        return outputs['regression_pred'], outputs['classification_probs']

class MultiTaskLoss(nn.Module):
    def __init__(self, regression_weight=1.0, classification_weight=0.5):
        super(MultiTaskLoss, self).__init__()
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, outputs, regression_targets, classification_targets):
        """
        计算多任务损失
        
        Args:
            outputs: 模型输出
            regression_targets: 回归目标 (价格)
            classification_targets: 分类目标 (涨跌方向)
        """
        # 回归损失
        regression_loss = self.mse_loss(outputs['regression_pred'], regression_targets)
        
        # 分类损失
        classification_loss = self.ce_loss(
            outputs['classification_logits'].view(-1, 3),
            classification_targets.view(-1)
        )
        
        # 总损失
        total_loss = (self.regression_weight * regression_loss + 
                     self.classification_weight * classification_loss)
        
        return {
            'total_loss': total_loss,
            'regression_loss': regression_loss,
            'classification_loss': classification_loss
        } 