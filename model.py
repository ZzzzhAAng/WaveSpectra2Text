from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Simple stack of BiLSTM layers operating over frame-wise features.
    Input: (batch, time, feature_dim)
    Output: (batch, time, hidden_dim * 2)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F)
        # lengths: (B,)
        # pack for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = self.layer_norm(out)
        return out, lengths


class ASRModel(nn.Module):
    """CTC acoustic model: Encoder + linear classifier.
    Produces frame-level logits and log-probs for CTC loss and decoding.
    """

    def __init__(
        self,
        num_classes: int,  # includes blank at index 0
        n_mels: int = 80,
        encoder_hidden: int = 256,
        encoder_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.encoder = Encoder(input_dim=n_mels, hidden_dim=encoder_hidden, num_layers=encoder_layers, dropout=dropout)
        self.classifier = nn.Linear(encoder_hidden * 2, num_classes)

    def forward(self, features: torch.Tensor, feature_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # features: (B, T, n_mels)
        enc_out, out_lengths = self.encoder(features, feature_lengths)
        logits = self.classifier(enc_out)  # (B, T, C)
        log_probs = F.log_softmax(logits, dim=-1)
        return logits, log_probs, out_lengths


class CTCGreedyDecoder:
    """Greedy decoder for CTC outputs.
    Expects log_probs of shape (B, T, C) and lengths (B,).
    """

    def __init__(self, blank_index: int = 0):
        self.blank_index = blank_index

    @torch.no_grad()
    def decode(self, log_probs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Argmax over classes -> (B, T)
        indices = log_probs.argmax(dim=-1)
        decoded = []
        for i in range(indices.size(0)):
            T = int(lengths[i].item())
            seq = indices[i, :T].tolist()
            # collapse repeats and remove blanks
            collapsed: list[int] = []
            prev = None
            for idx in seq:
                if idx == self.blank_index:
                    prev = idx
                    continue
                if prev is not None and idx == prev:
                    continue
                collapsed.append(idx)
                prev = idx
            decoded.append(torch.tensor(collapsed, dtype=torch.long))
        return decoded
