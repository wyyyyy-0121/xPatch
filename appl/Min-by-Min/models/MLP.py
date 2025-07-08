import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, seq_len, feature_dim, hidden_size, prediction_length, dropout=0.1, num_layers=5, use_layernorm=True):
        super().__init__()
        assert isinstance(seq_len, int) and seq_len > 0, f"seq_len必须为正整数，当前为{seq_len}"
        assert isinstance(feature_dim, int) and feature_dim > 0, f"feature_dim必须为正整数，当前为{feature_dim}"
        assert isinstance(prediction_length, int) and prediction_length > 0, f"prediction_length必须为正整数，当前为{prediction_length}"
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.prediction_length = prediction_length
        self.input_dim = seq_len * feature_dim
        self.output_dim = prediction_length * feature_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_layernorm = use_layernorm
        self.linears = nn.ModuleList([
            nn.Linear(self.input_dim if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_size, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() != 3 or x.size(1) != self.seq_len or x.size(2) != self.feature_dim:
            raise RuntimeError(f"MLPModel: 输入必须为[batch, {self.seq_len}, {self.feature_dim}]，实际为{x.shape}")
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        out = x
        for i in range(len(self.linears)):
            residual = out
            out = self.linears[i](out)
            if self.use_layernorm:
                out = self.norms[i](out)
            out = self.relu(out)
            out = self.dropouts[i](out)
            if out.shape == residual.shape:
                out = out + residual
        out = self.head(out)
        out = out.view(batch_size, self.prediction_length, self.feature_dim)
        return out
