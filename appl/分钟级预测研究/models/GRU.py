import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, prediction_length, dropout=0.1, use_layernorm=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.prediction_length = prediction_length
        self.input_dim = input_dim
        
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln = nn.LayerNorm(hidden_size)
        
        # 输出层：从hidden_size映射到prediction_length * input_dim
        self.fc = nn.Linear(hidden_size, prediction_length * input_dim)
        
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        out, _ = self.gru(x)
        
        if self.use_layernorm:
            out = self.ln(out)
        
        # 取最后一个时间步的输出
        last_output = out[:, -1, :]  # [batch, hidden_size]
        
        # 预测未来prediction_length步
        predictions = self.fc(last_output)  # [batch, prediction_length * input_dim]
        
        # 重塑为 [batch, prediction_length, input_dim]
        batch_size = predictions.size(0)
        predictions = predictions.view(batch_size, self.prediction_length, self.input_dim)
        
        return predictions
