import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, dropout=0.1, num_layers=5, use_layernorm=True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_layernorm = use_layernorm
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
            x = x.reshape(x.size(0), -1)
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
        return out
