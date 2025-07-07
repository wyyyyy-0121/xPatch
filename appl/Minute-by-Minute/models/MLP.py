import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, dropout=0.1, num_layers=5, use_layernorm=True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.use_layernorm = use_layernorm
        # 计算prediction_length（假设output_dim = prediction_length * feature_dim）
        self.prediction_length = output_dim  # 简化处理，实际使用时需要根据具体场景调整
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_size
            self.linears.append(nn.Linear(in_dim, hidden_size))
            self.norms.append(nn.LayerNorm(hidden_size))
            self.dropouts.append(nn.Dropout(dropout))
        self.head = nn.Linear(hidden_size, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        # x: [batch, seq_len, input_dim] or [batch, input_dim]
        if x.dim() == 3:
            batch_size = x.size(0)
            input_dim = x.size(2)
            seq_len = x.size(1)
            x = x.reshape(batch_size, -1)
        else:
            batch_size = x.size(0)
            input_dim = self.output_dim  # 兜底
        out = x
        for i in range(self.num_layers):
            residual = out
            out = self.linears[i](out)
            if self.use_layernorm:
                out = self.norms[i](out)
            out = self.relu(out)
            out = self.dropouts[i](out)
            if out.shape == residual.shape:
                out = out + residual  # 残差连接
        out = self.head(out)
        # 修正：输出reshape为[batch, prediction_length, input_dim]
        prediction_length = self.prediction_length
        feature_dim = input_dim if x.dim() == 2 else x.size(-1)
        out = out.view(batch_size, prediction_length, -1)
        return out
