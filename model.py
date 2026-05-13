"""PyTorch model for compact isolated-word lip reading."""

from __future__ import annotations

import torch
from torch import nn


class LipReadingModel(nn.Module):
    """CNN frame encoder + GRU temporal model + classifier head.

    Input shape: (batch, frames, channels, height, width).
    Each mouth frame is encoded by a small 2D CNN. The resulting sequence of
    frame embeddings is passed through a GRU so mouth motion over time can
    influence the final word prediction.
    """

    def __init__(
        self,
        num_classes: int,
        img_size: int = 96,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size

        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.temporal_model = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, frames, channels, height, width = x.shape
        x = x.reshape(batch_size * frames, channels, height, width)
        frame_features = self.frame_encoder(x)
        embeddings = self.embedding(frame_features)
        embeddings = embeddings.reshape(batch_size, frames, -1)

        gru_output, _ = self.temporal_model(embeddings)
        final_state = gru_output[:, -1, :]
        return self.classifier(final_state)
