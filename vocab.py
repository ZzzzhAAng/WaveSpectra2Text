import json
import os
from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional


BLANK_TOKEN = "<blank>"


def _unique_ordered_characters(texts: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for text in texts:
        if text is None:
            continue
        for ch in str(text):
            if ch not in seen:
                seen.add(ch)
                ordered.append(ch)
    return ordered


def normalize_text(text: str) -> str:
    """Minimal normalization for Chinese digits dataset.
    - Strip whitespace around
    - Replace consecutive spaces with a single space
    - Keep digits and CJK characters; other punctuation kept as-is (dataset-specific)
    """
    if text is None:
        return ""
    text = str(text).strip()
    # Collapse spaces
    text = " ".join(text.split())
    return text


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    blank_index: int = 0

    @classmethod
    def build_from_texts(cls, texts: Iterable[str], add_blank: bool = True) -> "Vocab":
        cleaned = [normalize_text(t) for t in texts if t is not None]
        charset = _unique_ordered_characters(cleaned)
        tokens: List[str] = []
        if add_blank:
            tokens.append(BLANK_TOKEN)
        tokens.extend(charset)
        stoi = {tok: i for i, tok in enumerate(tokens)}
        return cls(stoi=stoi, itos=tokens, blank_index=stoi[BLANK_TOKEN] if add_blank else -1)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls(stoi=obj["stoi"], itos=obj["itos"], blank_index=obj.get("blank_index", 0))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos, "blank_index": self.blank_index}, f, ensure_ascii=False, indent=2)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        text = normalize_text(text)
        return [self.stoi[ch] for ch in text if ch in self.stoi and ch != BLANK_TOKEN]

    def decode(self, ids: Iterable[int], collapse_repeats: bool = False, remove_blanks: bool = True) -> str:
        last_id: Optional[int] = None
        chars: List[str] = []
        for idx in ids:
            if idx < 0 or idx >= len(self.itos):
                continue
            if remove_blanks and idx == self.blank_index:
                last_id = idx
                continue
            if collapse_repeats and last_id == idx:
                continue
            chars.append(self.itos[idx])
            last_id = idx
        return "".join(chars)


def build_vocab_from_csv(csv_path: str, text_column: str = "text") -> Vocab:
    import pandas as pd
    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        # try common alternatives
        for candidate in ["label", "transcript", "sentence", "target", "y"]:
            if candidate in df.columns:
                text_column = candidate
                break
        else:
            raise ValueError(f"Text column not found in CSV. Available columns: {list(df.columns)}")
    texts = [normalize_text(x) for x in df[text_column].astype(str).tolist()]
    return Vocab.build_from_texts(texts)
