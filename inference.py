import argparse
import os
from typing import List, Tuple

import numpy as np
import torch

from model import ASRModel
from vocab import Vocab


def load_model(ckpt_path: str, device: torch.device) -> Tuple[ASRModel, Vocab]:
    state = torch.load(ckpt_path, map_location=device)
    hparams = state.get("model_hparams") or {}
    num_classes = int(hparams.get("num_classes"))
    n_mels = int(hparams.get("n_mels", 80))
    encoder_hidden = int(hparams.get("encoder_hidden", 256))
    encoder_layers = int(hparams.get("encoder_layers", 3))
    dropout = float(hparams.get("dropout", 0.1))

    model = ASRModel(
        num_classes=num_classes,
        n_mels=n_mels,
        encoder_hidden=encoder_hidden,
        encoder_layers=encoder_layers,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    # Load vocab
    vocab_path = state.get("vocab_path")
    if vocab_path is None or not os.path.exists(vocab_path):
        # fallback: look next to checkpoint
        cand = os.path.join(os.path.dirname(ckpt_path), "vocab.json")
        if os.path.exists(cand):
            vocab_path = cand
        else:
            raise FileNotFoundError("vocab.json not found; expected in checkpoint or alongside it")
    vocab = Vocab.load(vocab_path)
    return model, vocab


def normalize_per_sample(x: np.ndarray) -> np.ndarray:
    mu = float(x.mean())
    sigma = float(x.std())
    if sigma < 1e-5:
        sigma = 1.0
    return (x - mu) / sigma


def collect_npy_files(path: str) -> List[str]:
    files: List[str] = []
    if os.path.isdir(path):
        for root, _, fnames in os.walk(path):
            for f in fnames:
                if f.lower().endswith(".npy"):
                    files.append(os.path.join(root, f))
    else:
        files.append(path)
    files.sort()
    return files


@torch.no_grad()
def infer_files(model: ASRModel, vocab: Vocab, inputs: List[str], device: torch.device) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    for path in inputs:
        arr = np.load(path)
        # Accept either (T, F) or (F, T)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array in {path}, got shape {arr.shape}")
        T, F = arr.shape
        if T < F:
            # heuristically assume (F, T)
            arr = arr.T
        arr = arr.astype(np.float32)
        arr = normalize_per_sample(arr)
        feats = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, T, F)
        lengths = torch.tensor([feats.size(1)], dtype=torch.long).to(device)
        logits, log_probs, out_lengths = model(feats, lengths)
        # Greedy CTC decoding
        idx_seq = log_probs.argmax(dim=-1)[0, : out_lengths[0]].tolist()
        text = vocab.decode_from_indices(idx_seq, collapse_repeats=True, remove_blanks=True)
        results.append((path, text))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="ASR inference from spectrogram (.npy) only")
    parser.add_argument("--model", type=str, default="checkpoints/best.ckpt")
    parser.add_argument("--input", type=str, required=True, help="Path to .npy file or directory of .npy files")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]) 
    parser.add_argument("--output_csv", type=str, default="", help="Optional path to save results CSV")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, vocab = load_model(args.model, device)

    npy_files = collect_npy_files(args.input)
    results = infer_files(model, vocab, npy_files, device)

    for path, text in results:
        print(f"{path}\t{text}")

    if args.output_csv:
        import pandas as pd
        df = pd.DataFrame(results, columns=["path", "prediction"])
        df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
