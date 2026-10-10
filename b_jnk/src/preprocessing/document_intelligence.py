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


def _piece(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return ""
    for key in ("content", "text", "value"):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip()
    return ""


def _joined(items: Any, sep: str) -> str:
    if not isinstance(items, list):
        return ""
    parts = [_piece(item) for item in items]
    return sep.join(part for part in parts if part)


def page_text(page: dict, content: str) -> str:
    """One page. Older Read results use 'text'; current results use 'content'."""
    for key in ("content", "text"):
        own = page.get(key)
        if isinstance(own, str) and own.strip():
            return own.strip()
    lines = _joined(page.get("lines"), "\n")
    if lines:
        return lines
    words = _joined(page.get("words"), " ")
    if words:
        return words
    if not content:
        return ""
    chunks: list[str] = []
    for span in page.get("spans") or []:
        if not isinstance(span, dict):
            continue
        offset = int(span.get("offset") or 0)
        length = int(span.get("length") or 0)
        chunks.append(content[offset : offset + length])
    return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())


def _paragraphs_by_page(result: dict) -> dict[int, str]:
    grouped: dict[int, list[str]] = {}
    for para in result.get("paragraphs") or []:
        if not isinstance(para, dict):
            continue
        text = _piece(para)
        if not text:
            continue
        numbers = []
        for region in para.get("boundingRegions") or []:
            if isinstance(region, dict) and region.get("pageNumber"):
                numbers.append(int(region["pageNumber"]))
        for number in numbers or [1]:
            grouped.setdefault(number, []).append(text)
    return {number: "\n".join(parts) for number, parts in grouped.items()}


def _page_records(result: dict) -> list[tuple[int, dict]]:
    pages = result.get("pages")
    if isinstance(pages, dict):
        records = []
        for key, page in pages.items():
            if isinstance(page, dict):
                number = int(page.get("pageNumber") or page.get("page") or key)
                records.append((number, page))
        return records
    if isinstance(pages, list):
        records = []
        for index, page in enumerate(pages, start=1):
            if isinstance(page, dict):
                number = int(page.get("pageNumber") or page.get("page") or index)
                records.append((number, page))
        return records
    read_results = result.get("readResults")
    if isinstance(read_results, list):
        records = []
        for index, page in enumerate(read_results, start=1):
            if isinstance(page, dict):
                number = int(page.get("page") or page.get("pageNumber") or index)
                records.append((number, page))
        return records
    return []


def pages_in_result(result: dict) -> list[tuple[int, str]]:
    """(page number, text) in the order Document Intelligence numbered them."""
    content = _piece(result.get("content")) or _piece(result.get("text")) or _piece(result.get("markdown"))
    paragraphs = _paragraphs_by_page(result)
    found: list[tuple[int, str]] = []
    for number, page in _page_records(result):
        text = page_text(page, content) or paragraphs.get(number, "")
        found.append((number, text))
    if not found and content.strip():
        found.append((1, content.strip()))
    return found
