"""BERT page classifier: 8 labels (KEEP, BLANK, six JUNK subtypes).

torch / transformers are imported lazily so the TF-IDF path keeps working
without them installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

BERT_LABELS: tuple[str, ...] = (
    "KEEP",
    "BLANK",
    "JUNK_INVOICE",
    "JUNK_COVER_PAGE",
    "JUNK_RECORD_REQUEST",
    "JUNK_INSTRUCTIONS",
    "JUNK_LETTER_FAX",
    "JUNK_OTHERS",
)
META_FILE = "bjnk_meta.json"


def pick_device(preference: str = "auto"):
    import torch

    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode_head_tail(tokenizer, texts: list[str], max_length: int, head_share: float = 0.75) -> dict[str, Any]:
    """Tokenize keeping the start and end of long pages (headers and signatures both matter)."""
    import torch

    budget = max_length - 2
    head = int(budget * head_share)
    tail = budget - head
    start = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    end = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    ids_list = tokenizer(list(texts), add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
    seqs = []
    for ids in ids_list:
        if len(ids) > budget:
            ids = ids[:head] + ids[-tail:]
        seqs.append([start, *ids, end])
    width = max(len(s) for s in seqs)
    pad = tokenizer.pad_token_id
    input_ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    attention = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attention[i, : len(s)] = 1
    return {"input_ids": input_ids, "attention_mask": attention}


class BertPageClassifier:
    """Same interface as FlatClassifier where inference needs it."""

    bundle = None  # no TF-IDF coefficients to explain

    def __init__(self, model, tokenizer, *, labels: list[str], max_length: int, head_share: float, device, meta: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.labels = list(labels)
        self.max_length = max_length
        self.head_share = head_share
        self.device = device
        self.meta = meta
        self.name = f"bert_{meta.get('base_model', 'page')}"

    @classmethod
    def load(cls, path: str | Path, *, device: str = "auto") -> "BertPageClassifier":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(path)
        meta = json.loads((path / META_FILE).read_text(encoding="utf-8"))
        dev = pick_device(device)
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(dev)
        model.eval()
        return cls(
            model,
            tokenizer,
            labels=meta["labels"],
            max_length=int(meta.get("max_length", 512)),
            head_share=float(meta.get("head_share", 0.75)),
            device=dev,
            meta=meta,
        )

    def predict_proba(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = encode_head_tail(self.tokenizer, texts[i : i + batch_size], self.max_length, self.head_share)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits
                out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, len(self.labels)))

    def predict_labels(self, texts: list[str]) -> list[str]:
        proba = self.predict_proba(texts)
        return [self.labels[int(i)] for i in np.argmax(proba, axis=1)]


def save_bert(model, tokenizer, path: str | Path, meta: dict) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    (path / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def bert_available(path: str | Path) -> bool:
    path = Path(path)
    return (path / META_FILE).exists() and (path / "config.json").exists()
