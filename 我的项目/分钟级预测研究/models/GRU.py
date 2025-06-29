import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_dim, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
