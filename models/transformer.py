import torch
import torch.nn as nn
from typing import Optional


class SwathWidthTransformer(nn.Module):
    """
    Transformer encoder for temporal modeling of frame-level CNN features.

    Input:
        x: [B, T, D]

    Output:
        [B] if output_dim=1
        [B, output_dim] otherwise
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_frames: int = 25,
        num_layers: int = 2,
        num_heads: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        output_dim: int = 1,
        pooling: str = "mean",  # "mean" or "cls"
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.output_dim = output_dim
        self.pooling = pooling

        # --------------------------------------------------
        # Positional Encoding (Learnable)
        # --------------------------------------------------
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, num_frames, feature_dim)
        )

        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

        # Optional CLS token
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # --------------------------------------------------
        # Transformer Encoder
        # --------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # More stable training
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # --------------------------------------------------
        # Regression Head
        # --------------------------------------------------
        self.regressor = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    # ------------------------------------------------------
    # Weight Initialization
    # ------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """

        B, T, D = x.shape

        if T != self.num_frames:
            raise ValueError(
                f"Expected {self.num_frames} frames, got {T}"
            )

        # Add positional encoding
        x = x + self.positional_encoding

        # Optional CLS token pooling
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # Transformer
        x = self.transformer_encoder(x)

        # Pooling
        if self.pooling == "mean":
            x = x.mean(dim=1)
        elif self.pooling == "cls":
            x = x[:, 0]

        # Regression
        out = self.regressor(x)

        if self.output_dim == 1:
            return out.squeeze(-1)

        return out
