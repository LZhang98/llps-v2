import torch
import math

class Encoder (torch.nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, ff_dim, dropout=0.5) -> None:
        super().__init__()
        self.model_type = 'Encoder'
        self.pos_encoder = PositionalEncoding(model_dim, dropout)

        encoder_layers = torch.nn.TransformerEncoderLayer(model_dim, num_heads, ff_dim, dropout, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layers, num_layers)
    
    def forward(self, x):
        x = self.pos_encoder(x)
        return self.encoder(x)

    def extract_features(self, x):
        a = self.pos_encoder(x)
        b = self.encoder(a)

        return [a, b]

# TODO: complete image convolutional encoder
class ImageEncoder (torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_type = 'ImageEncoder'

        self.conv2d1 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2d2 = torch.nn.Conv2d(1, 1, 5)
        self.conv2d3 = torch.nn.Conv2d(1, 1, 3)

        self.encoder = torch.nn.Sequential(
            self.conv2d1,   # b, 128, 128, 1
            self.maxpool,   # b, 64, 64, 1
            self.conv2d2,
            self.maxpool,   # b, 32, 32, 1
            self.conv2d3,
            self.maxpool,   # b, 16, 16, 1
        )

        # b, 64
        self.flatten = torch.nn.Flatten(start_dim=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        return x

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