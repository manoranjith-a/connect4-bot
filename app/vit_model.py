import torch
import torch.nn as nn

class BoardViTStrong(nn.Module):
    """
    Conv-stem + Transformer (overlapping patches via conv)
    Input:  (B,2,6,7)
    Output: (B,7)
    """
    def __init__(self, d_model=256, nhead=8, num_layers=8, dim_ff=1024, dropout=0.10):
        super().__init__()
        self.num_tokens = 6 * 7  # 42

        self.stem = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, d_model, 3, padding=1),
            nn.GELU(),
        )

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, 1 + self.num_tokens, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 7),
        )

    def forward(self, x):
        B = x.size(0)
        x = self.stem(x)                                 # (B,d_model,6,7)
        x = x.permute(0, 2, 3, 1).reshape(B, 42, -1)      # (B,42,d_model)

        cls = self.cls.expand(B, -1, -1)                  # (B,1,d_model)
        x = torch.cat([cls, x], dim=1)                    # (B,43,d_model)
        x = x + self.pos

        x = self.encoder(x)
        x = self.norm(x)
        return self.head(x[:, 0])                         # CLS -> logits
