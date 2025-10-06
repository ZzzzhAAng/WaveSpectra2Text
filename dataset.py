from typing import Dict, List, Tuple, Optional
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from audio_features import load_audio, compute_log_mel_spectrogram
from vocab import Vocab, build_vocab_from_csv, normalize_text


class SpectrogramCTCDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        audio_dir: str,
        vocab: Optional[Vocab] = None,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        text_column: str = "text",
        path_column: str = "path",
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path)
        if path_column not in self.df.columns:
            # try common alternatives
            for candidate in ["audio", "file", "wav", "filename"]:
                if candidate in self.df.columns:
                    path_column = candidate
                    break
            else:
                raise ValueError(f"Audio path column not found. Available: {list(self.df.columns)}")
        self.path_column = path_column

        if text_column not in self.df.columns:
            for candidate in ["label", "transcript", "sentence", "target", "y"]:
                if candidate in self.df.columns:
                    text_column = candidate
                    break
            else:
                raise ValueError(f"Text column not found. Available: {list(self.df.columns)}")
        self.text_column = text_column

        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        if vocab is None:
            self.vocab = build_vocab_from_csv(csv_path, text_column=text_column)
        else:
            self.vocab = vocab

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        rel_path = str(row[self.path_column])
        text = normalize_text(str(row[self.text_column]))
        audio_path = rel_path if os.path.isabs(rel_path) else os.path.join(self.audio_dir, rel_path)

        wav, sr = load_audio(audio_path, target_sr=self.sample_rate)
        spec = compute_log_mel_spectrogram(
            wav=wav,
            sample_rate=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )  # [T, n_mels]
        target_ids = self.vocab.encode(text)

        return {
            "spectrogram": torch.tensor(spec, dtype=torch.float32),  # [T, n_mels]
            "targets": torch.tensor(target_ids, dtype=torch.int64),   # [L]
            "input_length": torch.tensor(spec.shape[0], dtype=torch.int64),
            "target_length": torch.tensor(len(target_ids), dtype=torch.int64),
            "text": text,
            "path": audio_path,
        }


def ctc_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Sort by input length (descending) for potential packing/efficiency
    batch = sorted(batch, key=lambda x: int(x["input_length"]), reverse=True)

    specs = [b["spectrogram"] for b in batch]
    targets = [b["targets"] for b in batch]
    input_lengths = torch.tensor([s.shape[0] for s in specs], dtype=torch.int64)
    target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.int64)

    max_T = int(input_lengths.max().item())
    n_mels = int(specs[0].shape[1])
    padded_specs = torch.zeros((len(batch), max_T, n_mels), dtype=torch.float32)
    for i, s in enumerate(specs):
        padded_specs[i, : s.shape[0]] = s

    # Concatenate targets for CTC
    concat_targets = torch.cat(targets, dim=0) if len(targets) > 0 else torch.empty((0,), dtype=torch.int64)

    return {
        "spectrogram": padded_specs.transpose(1, 2),  # [B, n_mels, T]
        "targets": concat_targets,                    # [sum(L)]
        "input_lengths": input_lengths,               # [B]
        "target_lengths": target_lengths,             # [B]
    }
