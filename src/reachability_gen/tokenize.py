"""Small vocab + integer tokenization for locked edge-list encodings.

Tokenizes ``encode_instance`` strings into integer ids for the feed-forward
transformer. Edge tokens ``u,v`` are split into ``u``, ``,``, ``v`` so the
vocab stays small (specials + decimal node ids).

RESEARCH / MEASURE plumbing only — no science OPEN claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

# Covers TRAIN_N / OOD_N (max 32) with headroom for future size-OOD.
DEFAULT_MAX_NODE_ID: int = 64

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS: tuple[str, ...] = (
    PAD_TOKEN,
    UNK_TOKEN,
    "N",
    "EDGES",
    "QUERY",
    ",",
)


@dataclass(frozen=True)
class Vocab:
    """Bidirectional string ↔ id map for reachability encodings."""

    stoi: dict[str, int]
    itos: tuple[str, ...]

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)

    def encode_token(self, tok: str) -> int:
        return self.stoi.get(tok, self.unk_id)

    def decode_id(self, idx: int) -> str:
        if 0 <= idx < len(self.itos):
            return self.itos[idx]
        return UNK_TOKEN


def build_vocab(*, max_node_id: int = DEFAULT_MAX_NODE_ID) -> Vocab:
    """Build a fixed vocab: specials + decimal strings ``0..max_node_id``."""
    if max_node_id < 0:
        raise ValueError(f"max_node_id must be >= 0, got {max_node_id}")
    itos: list[str] = list(SPECIAL_TOKENS)
    for i in range(max_node_id + 1):
        s = str(i)
        if s not in itos:
            itos.append(s)
    stoi = {t: i for i, t in enumerate(itos)}
    return Vocab(stoi=stoi, itos=tuple(itos))


def split_encoding_tokens(encoding: str) -> list[str]:
    """Whitespace-split encoding; expand ``u,v`` edge atoms into ``u``, ``,``, ``v``."""
    out: list[str] = []
    for tok in str(encoding).split():
        if "," in tok and tok not in (",",):
            # Canonical edge atom "u,v" (no spaces). Split for compact vocab.
            parts = tok.split(",")
            if len(parts) == 2 and parts[0] != "" and parts[1] != "":
                out.extend([parts[0], ",", parts[1]])
                continue
        out.append(tok)
    return out


def encode_to_ids(encoding: str, vocab: Vocab) -> list[int]:
    """Map one encoding string to a list of token ids."""
    return [vocab.encode_token(t) for t in split_encoding_tokens(encoding)]


def pad_batch(
    sequences: Sequence[Sequence[int]],
    *,
    pad_id: int,
    max_len: int | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """Pad to rectangular batch; return (token_ids, attention_mask) with 1=real."""
    if not sequences:
        return [], []
    lengths = [len(s) for s in sequences]
    target = max(lengths) if max_len is None else int(max_len)
    if target < 1:
        target = 1
    ids: list[list[int]] = []
    mask: list[list[int]] = []
    for seq in sequences:
        truncated = list(seq[:target])
        pad_n = target - len(truncated)
        ids.append(truncated + [pad_id] * pad_n)
        mask.append([1] * len(truncated) + [0] * pad_n)
    return ids, mask


def batch_encode(
    encodings: Iterable[str],
    vocab: Vocab,
    *,
    max_len: int | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """Encode + pad a batch of encoding strings."""
    seqs = [encode_to_ids(e, vocab) for e in encodings]
    return pad_batch(seqs, pad_id=vocab.pad_id, max_len=max_len)


__all__ = [
    "DEFAULT_MAX_NODE_ID",
    "PAD_TOKEN",
    "SPECIAL_TOKENS",
    "UNK_TOKEN",
    "Vocab",
    "batch_encode",
    "build_vocab",
    "encode_to_ids",
    "pad_batch",
    "split_encoding_tokens",
]
