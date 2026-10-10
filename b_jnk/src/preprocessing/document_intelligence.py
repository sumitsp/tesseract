"""Page text from an Azure Document Intelligence analyze result.

A file may be one page or a whole chart. Text is the reading-order lines on
that page. The full-document string is used only when a page has no lines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def iter_analyze_results(payload: Any) -> Iterator[dict]:
    """Yield each analyze result, with or without the service envelope."""
    if isinstance(payload, list):
        for item in payload:
            yield from iter_analyze_results(item)
        return
    if not isinstance(payload, dict):
        return
    for key in ("analyzeResult", "result"):
        inner = payload.get(key)
        if isinstance(inner, dict) and ("pages" in inner or "content" in inner):
            yield inner
            return
    if isinstance(payload.get("pages"), list):
        yield payload


def page_text(page: dict, content: str) -> str:
    lines = page.get("lines") or []
    if lines:
        parts = [str(line.get("content") or "").strip() for line in lines]
        return "\n".join(part for part in parts if part)
    words = page.get("words") or []
    if words:
        parts = [str(word.get("content") or "").strip() for word in words]
        return " ".join(part for part in parts if part)
    if not content:
        return ""
    chunks: list[str] = []
    for span in page.get("spans") or []:
        offset = int(span.get("offset") or 0)
        length = int(span.get("length") or 0)
        chunks.append(content[offset : offset + length])
    return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())


def pages_in_result(result: dict) -> list[tuple[int, str]]:
    """(page number, text) in the order Document Intelligence numbered them."""
    content = str(result.get("content") or "")
    pages = result.get("pages") or []
    found: list[tuple[int, str]] = []
    for index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            continue
        number = int(page.get("pageNumber") or index)
        found.append((number, page_text(page, content)))
    if not found and content.strip():
        found.append((1, content.strip()))
    return found
