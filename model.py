from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSubsampler(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, n_mels: int, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.subsample = ConvSubsampler(n_mels, hidden_dim)
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, n_mels, T]
        x = self.subsample(x)  # [B, H, T/4]
        x = x.transpose(1, 2)  # [B, T', H]
        out, _ = self.rnn(x)   # [B, T', 2H]
        out = self.proj(out)   # [B, T', H]
        return out, torch.tensor(out.shape[1], device=out.device).repeat(out.shape[0])


class CTCDecoder(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, enc_out: torch.Tensor) -> torch.Tensor:
        # enc_out: [B, T, H]
        logits = self.fc(enc_out)  # [B, T, V]
        return logits


class SpectrogramCTCModel(nn.Module):
    def __init__(self, n_mels: int, vocab_size: int, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = Encoder(n_mels=n_mels, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.decoder = CTCDecoder(hidden_dim=hidden_dim, vocab_size=vocab_size)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: [B, n_mels, T]
        enc_out, _ = self.encoder(spec)
        logits = self.decoder(enc_out)  # [B, T', V]
        return logits

    @torch.no_grad()
    def greedy_decode(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: [1, n_mels, T]
        self.eval()
        logits = self.forward(spec)  # [1, T', V]
        probs = F.log_softmax(logits, dim=-1)
        ids = probs.argmax(dim=-1).squeeze(0)  # [T']
        return ids
