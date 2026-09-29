"""Docling page content extraction — structure-based, no magic OCR strings.

Absolute-blank candidates are detected from empty DoclingDocument arrays
(texts / pictures / tables / forms / body children), not by matching
phrases like \"[no text detected]\".
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DoclingPageContent:
    text: str
    n_texts: int
    n_pictures: int
    n_tables: int
    n_forms: int
    n_key_values: int
    n_body_children: int
    is_structurally_empty: bool
    has_pictures: bool

    def to_meta(self) -> dict[str, Any]:
        return asdict(self)


_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u2028", "\n").replace("\u2029", "\n").replace("\xa0", " ")
    text = _HTML_COMMENT.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _len_list(obj: Any, key: str) -> int:
    val = obj.get(key) if isinstance(obj, dict) else None
    return len(val) if isinstance(val, list) else 0


def _body_children_count(doc: dict) -> int:
    body = doc.get("body")
    if not isinstance(body, dict):
        return 0
    children = body.get("children")
    return len(children) if isinstance(children, list) else 0


def _text_from_docling_texts(doc: dict) -> str:
    """Join real text nodes from document.texts (ignore empty/missing)."""
    texts = doc.get("texts")
    if not isinstance(texts, list):
        return ""
    parts: list[str] = []
    for node in texts:
        if isinstance(node, str) and node.strip():
            parts.append(node.strip())
            continue
        if not isinstance(node, dict):
            continue
        for key in ("text", "orig", "content", "value"):
            val = node.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
                break
    return _clean_text("\n".join(parts))


def extract_docling_page(page_obj: Any) -> DoclingPageContent:
    """Extract classifier text + structural emptiness from one Docling page object.

    Expected shape::
        {"markdown": "...", "document": { "texts": [], "pictures": [], ... }}
    """
    if page_obj is None:
        return DoclingPageContent(
            text="",
            n_texts=0,
            n_pictures=0,
            n_tables=0,
            n_forms=0,
            n_key_values=0,
            n_body_children=0,
            is_structurally_empty=True,
            has_pictures=False,
        )

    if isinstance(page_obj, str):
        text = _clean_text(page_obj)
        empty = not bool(text)
        return DoclingPageContent(
            text=text,
            n_texts=1 if text else 0,
            n_pictures=0,
            n_tables=0,
            n_forms=0,
            n_key_values=0,
            n_body_children=0,
            is_structurally_empty=empty,
            has_pictures=False,
        )

    if not isinstance(page_obj, dict):
        text = _clean_text(str(page_obj))
        return DoclingPageContent(
            text=text,
            n_texts=1 if text else 0,
            n_pictures=0,
            n_tables=0,
            n_forms=0,
            n_key_values=0,
            n_body_children=0,
            is_structurally_empty=not bool(text),
            has_pictures=False,
        )

    doc = page_obj.get("document")
    has_doc = isinstance(doc, dict)

    if has_doc:
        n_texts = _len_list(doc, "texts")
        n_pictures = _len_list(doc, "pictures")
        n_tables = _len_list(doc, "tables")
        n_forms = _len_list(doc, "form_items")
        n_kv = _len_list(doc, "key_value_items")
        n_body = _body_children_count(doc)
        text = _text_from_docling_texts(doc)
        # If texts array empty, do not trust placeholder markdown — structure wins
        if not text and isinstance(page_obj.get("markdown"), str) and n_texts > 0:
            text = _clean_text(page_obj["markdown"])
        structurally_empty = (
            n_texts == 0
            and n_pictures == 0
            and n_tables == 0
            and n_forms == 0
            and n_kv == 0
            and n_body == 0
        )
        return DoclingPageContent(
            text=text,
            n_texts=n_texts,
            n_pictures=n_pictures,
            n_tables=n_tables,
            n_forms=n_forms,
            n_key_values=n_kv,
            n_body_children=n_body,
            is_structurally_empty=structurally_empty,
            has_pictures=n_pictures > 0,
        )

    # No document block — fall back to markdown/ocr_text fields only
    text = ""
    for key in ("ocr_text", "text", "content", "raw_text", "markdown"):
        val = page_obj.get(key)
        if isinstance(val, str) and val.strip():
            text = _clean_text(val)
            if text:
                break
    return DoclingPageContent(
        text=text,
        n_texts=1 if text else 0,
        n_pictures=0,
        n_tables=0,
        n_forms=0,
        n_key_values=0,
        n_body_children=0,
        is_structurally_empty=not bool(text),
        has_pictures=False,
    )
