import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, prediction_length, dropout=0.1, use_layernorm=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.prediction_length = prediction_length
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, prediction_length * input_dim)
        # 新增：用于记录调试统计量
        self.debug_stats = {"out_mean": [], "out_std": [], "pred_mean": [], "pred_std": []}

    def forward(self, x):
        assert x.dim() == 3, f"LSTMModel expects 3D input, got {x.shape}"
        out, _ = self.lstm(x)
        if self.use_layernorm:
            out = self.ln(out)
        last_output = out[:, -1, :]
        predictions = self.fc(last_output)
        batch_size = predictions.size(0)
        predictions = predictions.view(batch_size, self.prediction_length, self.input_dim)
        # 新增：记录均值/方差
        self.debug_stats["out_mean"].append(out.mean().item())
        self.debug_stats["out_std"].append(out.std().item())
        self.debug_stats["pred_mean"].append(predictions.mean().item())
        self.debug_stats["pred_std"].append(predictions.std().item())
        return predictions
