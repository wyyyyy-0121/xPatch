import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_dim, dropout=0.1, use_layernorm=True):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_dim)
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        out, _ = self.gru(x)
        if self.use_layernorm:
            out = self.ln(out)
        out = self.fc(out)  # [batch, seq_len, output_dim]
        return out
