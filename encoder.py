import torch
import math

class Encoder (torch.nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, ff_dim, dropout=0.5) -> None:
        super().__init__()
        self.model_type = 'Encoder'
        self.pos_encoder = PositionalEncoding(model_dim, dropout)

        encoder_layers = torch.nn.TransformerEncoderLayer(model_dim, num_heads, ff_dim, dropout)
        self.encoder = torch.nn.TransformerEncoder(encoder_layers, num_layers)
    
    def forward(self, x):
        x = self.pos_encoder(x)
        return self.encoder(x)


class PositionalEncoding (torch.nn.Module):
    def __init__(self, model_dim, dropout=0.5, max_len=5000) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(max_len, 1, model_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)