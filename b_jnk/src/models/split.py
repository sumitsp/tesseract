"""Group-aware train/validation split by template/document."""

from __future__ import annotations

import random
from collections import Counter, defaultdict

from src.preprocessing.dataset import PageRecord


def group_train_test_split(
    rows: list[PageRecord],
    *,
    test_size: float = 0.25,
    random_state: int = 42,
    max_pages_per_template_train: int | None = 3,
) -> tuple[list[PageRecord], list[PageRecord]]:
    """Hold out whole split_group values; optionally cap train pages/template."""
    by_group: dict[str, list[PageRecord]] = defaultdict(list)
    for r in rows:
        by_group[r.group_key].append(r)

    groups = sorted(by_group.keys())
    rng = random.Random(random_state)
    rng.shuffle(groups)

    n_test_groups = max(1, int(round(len(groups) * test_size))) if len(groups) > 1 else 0
    test_groups = set(groups[:n_test_groups])
    train_groups = set(groups[n_test_groups:]) or set(groups) - test_groups

    train: list[PageRecord] = []
    test: list[PageRecord] = []
    for g, pages in by_group.items():
        if g in test_groups:
            test.extend(pages)
        else:
            pages_sorted = sorted(pages, key=lambda p: p.page_id)
            if max_pages_per_template_train is not None:
                pages_sorted = pages_sorted[: max_pages_per_template_train]
            train.extend(pages_sorted)

    if not train:
        # Degenerate tiny set: put everything in train for smoke fitting.
        train = list(rows)
        test = []
    return train, test


def describe_split(train: list[PageRecord], test: list[PageRecord]) -> dict:
    return {
        "train_pages": len(train),
        "test_pages": len(test),
        "train_groups": len({r.group_key for r in train}),
        "test_groups": len({r.group_key for r in test}),
        "train_classes": dict(Counter(r.primary_class for r in train)),
        "test_classes": dict(Counter(r.primary_class for r in test)),
        "train_flags": dict(Counter(r.flag for r in train if r.flag)),
        "test_flags": dict(Counter(r.flag for r in test if r.flag)),
        "overlap_groups": sorted(
            {r.group_key for r in train} & {r.group_key for r in test}
        ),
    }
