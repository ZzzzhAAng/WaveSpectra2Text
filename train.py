import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import ASRModel, CTCGreedyDecoder
from vocab import Vocab


TARGET_SR = 16000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    data_dir: str = "data"
    labels_csv: str = "data/labels.csv"
    output_dir: str = "checkpoints"
    spec_cache_dir: str = "data/specs"
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 160  # 10ms at 16kHz
    win_length: int = 400  # 25ms at 16kHz
    fmin: int = 20
    fmax: int = 7600
    encoder_hidden: int = 256
    encoder_layers: int = 3
    dropout: float = 0.1
    grad_clip: float = 5.0
    val_size: float = 0.15


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_labels(labels_csv: str, base_dir: str) -> List[Tuple[str, str]]:
    df = pd.read_csv(labels_csv)
    # Try to find audio and text columns flexibly
    audio_cols = [
        c
        for c in df.columns
        if c.lower() in {"path", "audio", "wav", "file", "filepath", "relative_path"}
    ]
    text_cols = [c for c in df.columns if c.lower() in {"text", "label", "transcript", "sentence"}]
    if not audio_cols or not text_cols:
        # Fall back to first two columns
        audio_col = df.columns[0]
        text_col = df.columns[1]
    else:
        audio_col = audio_cols[0]
        text_col = text_cols[0]
    pairs: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        rel = str(row[audio_col])
        text = str(row[text_col]) if not pd.isna(row[text_col]) else ""
        # Join with base_dir if path not absolute
        if not os.path.isabs(rel):
            rel = os.path.join(base_dir, rel)
        pairs.append((rel, text))
    return pairs


def compute_log_mel(path: str, cfg: TrainConfig) -> np.ndarray:
    # Load audio and resample to TARGET_SR
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    # Mel-spectrogram (power)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=TARGET_SR,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
    )
    # Log scale
    S = np.log(S + 1e-6)
    # (n_mels, time) -> (time, n_mels)
    S = S.T.astype(np.float32)
    return S


def normalize_per_sample(x: np.ndarray) -> np.ndarray:
    mu = float(x.mean())
    sigma = float(x.std())
    if sigma < 1e-5:
        sigma = 1.0
    return (x - mu) / sigma


class AudioLabelDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], vocab: Vocab, cfg: TrainConfig, use_cache: bool = True) -> None:
        self.pairs = pairs
        self.vocab = vocab
        self.cfg = cfg
        self.use_cache = use_cache
        if use_cache:
            ensure_dir(cfg.spec_cache_dir)

    def __len__(self) -> int:
        return len(self.pairs)

    def _spec_cache_path(self, audio_path: str) -> str:
        # Derive stable .npy path inside cache dir, mirroring folder structure where possible
        base = os.path.splitext(os.path.basename(audio_path))[0]
        return os.path.join(self.cfg.spec_cache_dir, base + f"_mel{self.cfg.n_mels}.npy")

    def __getitem__(self, idx: int):
        audio_path, text = self.pairs[idx]
        if self.use_cache:
            spec_path = self._spec_cache_path(audio_path)
            if os.path.exists(spec_path):
                feat = np.load(spec_path)
            else:
                feat = compute_log_mel(audio_path, self.cfg)
                feat = normalize_per_sample(feat)
                np.save(spec_path, feat)
        else:
            feat = compute_log_mel(audio_path, self.cfg)
            feat = normalize_per_sample(feat)
        labels = self.vocab.encode(text)
        feat_tensor = torch.from_numpy(feat)  # (T, F)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        return feat_tensor, label_tensor


def collate_fn(batch):
    feats, labels = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in feats], dtype=torch.long)
    feat_dim = feats[0].size(1)
    max_len = int(lengths.max().item())
    feats_padded = torch.zeros((len(feats), max_len, feat_dim), dtype=torch.float32)
    for i, f in enumerate(feats):
        T = f.size(0)
        feats_padded[i, :T] = f
    # concat targets for CTC
    targets = torch.cat(labels, dim=0)
    target_lengths = torch.tensor([l.size(0) for l in labels], dtype=torch.long)
    return feats_padded, lengths, targets, target_lengths


def compute_cer(ref: str, hyp: str) -> float:
    # Levenshtein distance over characters
    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    if n == 0:
        return float(m)
    return dp[n][m] / max(1, n)


def train_epoch(model: ASRModel, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.CTCLoss, device: torch.device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    total_frames = 0
    for feats, feat_lengths, targets, target_lengths in tqdm(loader, desc="train", leave=False):
        feats = feats.to(device)
        feat_lengths = feat_lengths.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, log_probs, out_lengths = model(feats, feat_lengths)
        # CTC expects (T, N, C)
        log_probs_tnc = log_probs.permute(1, 0, 2).contiguous()
        loss = criterion(log_probs_tnc, targets, out_lengths, target_lengths)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * feats.size(0)
        total_frames += int(out_lengths.sum().item())
    avg_loss = running_loss / max(1, len(loader.dataset))
    avg_frames = total_frames / max(1, len(loader.dataset))
    return avg_loss, avg_frames


@torch.no_grad()
def validate_epoch(model: ASRModel, loader: DataLoader, decoder: CTCGreedyDecoder, vocab: Vocab, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_cer = 0.0
    count = 0
    running_loss = 0.0
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    for feats, feat_lengths, targets, target_lengths in tqdm(loader, desc="valid", leave=False):
        feats = feats.to(device)
        feat_lengths = feat_lengths.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        logits, log_probs, out_lengths = model(feats, feat_lengths)
        log_probs_tnc = log_probs.permute(1, 0, 2).contiguous()
        loss = criterion(log_probs_tnc, targets, out_lengths, target_lengths)
        running_loss += loss.item() * feats.size(0)

        dec = decoder.decode(log_probs, out_lengths)
        # reconstruct targets per-sample to compute CER
        offset = 0
        for i in range(feats.size(0)):
            L = int(target_lengths[i].item())
            tgt_indices = targets[offset : offset + L].tolist()
            offset += L
            ref = vocab.decode_from_indices(tgt_indices, collapse_repeats=False, remove_blanks=True)
            hyp = vocab.decode_from_indices(dec[i].tolist(), collapse_repeats=False, remove_blanks=True)
            total_cer += compute_cer(ref, hyp)
            count += 1
    avg_cer = total_cer / max(1, count)
    avg_loss = running_loss / max(1, len(loader.dataset))
    return avg_loss, avg_cer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CTC ASR from spectrograms")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--labels_csv", type=str, default="data/labels.csv")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--spec_cache_dir", type=str, default="data/specs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--resume", type=str, default="", help="path to checkpoint to resume")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        spec_cache_dir=args.spec_cache_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        n_mels=args.n_mels,
    )

    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.spec_cache_dir)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = read_labels(cfg.labels_csv, base_dir=cfg.data_dir)
    # Split train/val on utterance level
    train_pairs, val_pairs = train_test_split(pairs, test_size=cfg.val_size, random_state=cfg.seed)

    # Build vocab from training references only
    vocab = Vocab.build_from_sentences([t for _, t in train_pairs])

    # Datasets
    train_ds = AudioLabelDataset(train_pairs, vocab=vocab, cfg=cfg, use_cache=True)
    val_ds = AudioLabelDataset(val_pairs, vocab=vocab, cfg=cfg, use_cache=True)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True)

    # Model
    model = ASRModel(
        num_classes=vocab.num_classes_for_ctc,
        n_mels=cfg.n_mels,
        encoder_hidden=cfg.encoder_hidden,
        encoder_layers=cfg.encoder_layers,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    decoder = CTCGreedyDecoder(blank_index=0)

    best_val_cer = math.inf
    best_ckpt_path = os.path.join(cfg.output_dir, "best.ckpt")

    # Optionally resume
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optim_state"])
        start_epoch = int(state.get("epoch", 0))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # Save vocab now for inference
    vocab_path = os.path.join(cfg.output_dir, "vocab.json")
    vocab.save(vocab_path)

    for epoch in range(start_epoch, cfg.epochs):
        print(f"Epoch {epoch+1}/{cfg.epochs}")
        train_loss, _ = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_cer = validate_epoch(model, val_loader, decoder, vocab, device)
        scheduler.step(val_cer)

        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_CER={val_cer:.4f}")

        # Save best by CER
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            payload = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "config": asdict(cfg),
                "vocab_path": vocab_path,
                "metric": {"val_cer": best_val_cer, "val_loss": val_loss},
                "model_hparams": {
                    "num_classes": vocab.num_classes_for_ctc,
                    "n_mels": cfg.n_mels,
                    "encoder_hidden": cfg.encoder_hidden,
                    "encoder_layers": cfg.encoder_layers,
                    "dropout": cfg.dropout,
                },
            }
            torch.save(payload, best_ckpt_path)
            print(f"  Saved best checkpoint to {best_ckpt_path}")

    print("Training complete.")
    print(f"Best CER: {best_val_cer:.4f}")


if __name__ == "__main__":
    main()
