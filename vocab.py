import json
from typing import Dict, List, Iterable


class Vocab:
    """Character-level vocabulary with CTC blank at index 0.
    Persistable to JSON. Builds from training transcripts.
    """

    def __init__(self, token_to_index: Dict[str, int], blank_index: int = 0):
        if blank_index != 0:
            raise ValueError("blank_index must be 0 for CTC")
        self.blank_index = blank_index
        self.token_to_index = dict(token_to_index)
        self.index_to_token = {idx: tok for tok, idx in self.token_to_index.items()}
        if self.blank_index not in self.index_to_token:
            # reserve 0 for blank
            self.index_to_token[self.blank_index] = "<blank>"
        if "<blank>" not in self.token_to_index:
            self.token_to_index["<blank>"] = self.blank_index

    @staticmethod
    def build_from_sentences(sentences: Iterable[str]) -> "Vocab":
        tokens: List[str] = []
        unique = set()
        for s in sentences:
            if s is None:
                continue
            s = str(s)
            # remove spaces; treat each Unicode char as token
            s = s.replace(" ", "")
            for ch in s:
                if ch not in unique and ch != "":
                    unique.add(ch)
                    tokens.append(ch)
        # deterministic order: sort by Unicode codepoint
        tokens = sorted(tokens)
        token_to_index = {"<blank>": 0}
        # tokens start at 1 to reserve 0 for blank
        for i, tok in enumerate(tokens, start=1):
            token_to_index[tok] = i
        return Vocab(token_to_index)

    def __len__(self) -> int:
        # number of non-blank tokens
        return max(self.token_to_index.values())

    @property
    def num_classes_for_ctc(self) -> int:
        # classes = vocab_size + 1 (blank)
        return len(self.token_to_index)

    def encode(self, text: str) -> List[int]:
        text = (text or "").replace(" ", "")
        indices: List[int] = []
        for ch in text:
            if ch not in self.token_to_index:
                raise KeyError(f"Unknown token '{ch}' not in vocabulary")
            indices.append(self.token_to_index[ch])
        return indices

    def decode_from_indices(self, indices: Iterable[int], collapse_repeats: bool = True, remove_blanks: bool = True) -> str:
        decoded: List[str] = []
        prev: int = None  # type: ignore
        for idx in indices:
            if remove_blanks and idx == self.blank_index:
                prev = idx
                continue
            if collapse_repeats and prev is not None and idx == prev:
                continue
            tok = self.index_to_token.get(idx)
            if tok is None or tok == "<blank>":
                prev = idx
                continue
            decoded.append(tok)
            prev = idx
        return "".join(decoded)

    def save(self, path: str) -> None:
        obj = {
            "blank_index": self.blank_index,
            "tokens": [self.index_to_token[i] for i in range(len(self.token_to_index))],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        tokens: List[str] = obj["tokens"]
        if tokens[0] != "<blank>":
            # normalize just in case
            tokens = ["<blank>"] + [t for t in tokens if t != "<blank>"]
        token_to_index = {tok: i for i, tok in enumerate(tokens)}
        return Vocab(token_to_index, blank_index=0)
