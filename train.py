import argparse
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SpectrogramCTCDataset, ctc_collate_fn
from vocab import Vocab, build_vocab_from_csv
from model import SpectrogramCTCModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train CTC spectrogram-to-text model")
    parser.add_argument("--csv", type=str, default="data/labels.csv", help="Path to labels CSV")
    parser.add_argument("--audio_dir", type=str, default="data/audio", help="Directory with audio files")
    parser.add_argument("--text_col", type=str, default="text", help="Text column name in CSV")
    parser.add_argument("--path_col", type=str, default="path", help="Audio path column name in CSV")

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")

    return parser.parse_args()


def train():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Build vocab from CSV
    vocab = build_vocab_from_csv(args.csv, text_column=args.text_col)

    # Dataset and loader
    ds = SpectrogramCTCDataset(
        csv_path=args.csv,
        audio_dir=args.audio_dir,
        vocab=vocab,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        text_column=args.text_col,
        path_column=args.path_col,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=ctc_collate_fn,
        pin_memory=True,
    )

    model = SpectrogramCTCModel(n_mels=args.n_mels, vocab_size=len(vocab), hidden_dim=args.hidden_dim, num_layers=args.layers, dropout=args.dropout)
    model = model.to(args.device)

    criterion = nn.CTCLoss(blank=vocab.blank_index, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.device.startswith("cuda")))

    start_epoch = 1
    global_step = 0

    if args.resume is not None and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(state["model"]) 
        optimizer.load_state_dict(state["optim"]) 
        scaler.load_state_dict(state.get("scaler", scaler.state_dict()))
        start_epoch = state.get("epoch", 0) + 1
        global_step = state.get("global_step", 0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            specs = batch["spectrogram"].to(args.device)       # [B, C, T]
            targets = batch["targets"].to(args.device)          # [sum(L)]
            input_lengths = batch["input_lengths"].to(args.device)
            target_lengths = batch["target_lengths"].to(args.device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.device.startswith("cuda"))):
                logits = model(specs)                # [B, T', V]
                log_probs = logits.log_softmax(dim=-1)
                log_probs = log_probs.transpose(0, 1)  # [T', B, V] for CTCLoss
                loss = criterion(log_probs, targets, input_lengths // 4, target_lengths)
                # We used 2-stride conv twice -> factor of 4 subsampling

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            global_step += 1
            pbar.set_postfix({"loss": f"{running_loss / global_step:.4f}"})

        ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "vocab": {"stoi": vocab.stoi, "itos": vocab.itos, "blank_index": vocab.blank_index},
                "config": vars(args),
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    train()
