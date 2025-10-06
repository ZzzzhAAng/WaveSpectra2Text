import argparse
import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from audio_features import load_audio, compute_log_mel_spectrogram
from vocab import Vocab
from model import SpectrogramCTCModel


def parse_args():
    parser = argparse.ArgumentParser(description="Greedy inference for CTC spectrogram-to-text model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint .pt")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", type=str, help="Path to input audio file")
    group.add_argument("--spectrogram", type=str, help="Path to precomputed spectrogram .npy [T, n_mels]")

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    state = torch.load(args.checkpoint, map_location=args.device)
    vocab = Vocab(stoi=state["vocab"]["stoi"], itos=state["vocab"]["itos"], blank_index=state["vocab"].get("blank_index", 0))

    model = SpectrogramCTCModel(n_mels=args.n_mels, vocab_size=len(vocab), hidden_dim=state["config"]["hidden_dim"], num_layers=state["config"]["layers"], dropout=state["config"]["dropout"]).to(args.device)
    model.load_state_dict(state["model"])  
    model.eval()

    if args.spectrogram is not None:
        spec = np.load(args.spectrogram)
        if spec.ndim != 2 or spec.shape[1] != args.n_mels:
            raise ValueError(f"Spectrogram shape must be [T, {args.n_mels}], got {spec.shape}")
    else:
        wav, sr = load_audio(args.audio, target_sr=args.sample_rate)
        spec = compute_log_mel_spectrogram(
            wav=wav,
            sample_rate=sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
        )  # [T, n_mels]

    spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).transpose(1, 2).to(args.device)  # [1, n_mels, T]

    with torch.no_grad():
        logits = model(spec_tensor)  # [1, T', V]
        log_probs = F.log_softmax(logits, dim=-1)
        ids = log_probs.argmax(dim=-1).squeeze(0).tolist()  # [T']
    # CTC collapse
    collapsed = []
    prev = None
    for i in ids:
        if i == vocab.blank_index:
            prev = i
            continue
        if prev == i:
            continue
        collapsed.append(i)
        prev = i
    text = vocab.decode(collapsed, collapse_repeats=False, remove_blanks=True)
    print(text)


if __name__ == "__main__":
    main()
