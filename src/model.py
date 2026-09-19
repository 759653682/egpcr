import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class GlobalTransformer(nn.Module):
    def __init__(
        self,
        window_size,
        num_global_features,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        dropout=0.1,
    ):
        super().__init__()
        self.window_size = window_size
        self.input_embedding = nn.Linear(num_global_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=window_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.decoder = nn.Linear(d_model, num_global_features)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def encode(self, x_window):
        embedded_x = self.input_embedding(x_window)
        pos_encoded_x = embedded_x.permute(1, 0, 2)
        pos_encoded_x = self.pos_encoder(pos_encoded_x)
        pos_encoded_x = pos_encoded_x.permute(1, 0, 2)
        return self.transformer_encoder(pos_encoded_x)

    def forward(self, x_window, target_window=None, return_embedding=False):
        if target_window is None:
            target_window = x_window

        encoder_output = self.encode(x_window)
        reconstructed = self.decoder(encoder_output)

        if self.training:
            return F.mse_loss(reconstructed, target_window)

        spe_loss = F.mse_loss(
            reconstructed, target_window, reduction="none"
        ).mean(dim=[1, 2])

        if return_embedding:
            embedding = torch.mean(encoder_output, dim=1)
            return spe_loss, embedding

        return spe_loss
