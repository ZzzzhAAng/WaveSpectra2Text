from typing import Tuple, Optional
import numpy as np
import librosa


def compute_log_mel_spectrogram(
    wav: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: Optional[int] = None,
    n_mels: int = 80,
    fmin: float = 20.0,
    fmax: Optional[float] = None,
    ref_level_db: float = 20.0,
    min_level_db: float = -100.0,
    power: float = 2.0,
    eps: float = 1e-10,
) -> np.ndarray:
    """Compute log-mel spectrogram from mono waveform.

    Returns array of shape [num_frames, n_mels].
    """
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if win_length is None:
        win_length = n_fft

    mel_filter = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True, window="hann")
    magnitude = np.abs(stft) ** power
    mel_spec = np.dot(mel_filter, magnitude)

    log_mel = 20.0 * np.log10(np.maximum(eps, mel_spec))
    log_mel = np.clip(log_mel - ref_level_db, min_level_db, 0)
    # Normalize to [0, 1]
    log_mel = (log_mel - min_level_db) / (-min_level_db)

    return log_mel.T  # [T, n_mels]


def load_audio(path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Load audio as float32 mono waveform in [-1, 1].
    If target_sr is provided, resample to that sample rate.
    """
    wav, sr = librosa.load(path, sr=None, mono=True)
    if (target_sr is not None) and (sr != target_sr):
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    wav = wav.astype(np.float32)
    return wav, sr
