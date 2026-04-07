from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence, TypedDict


class HoldoutSplit(TypedDict):
    train_texts: list[str]
    val_texts: list[str]
    test_texts: list[str]
    y_train: list[int]
    y_val: list[int]
    y_test: list[int]


def _split_counts(n: int, test_size: float, val_size: float) -> tuple[int, int, int]:
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_train = n - n_test - n_val
    if n_train <= 0:
        raise ValueError(
            f"Not enough samples per class after split: n={n}, "
            f"test_size={test_size}, val_size={val_size}"
        )
    return n_train, n_val, n_test


def prepare_holdout_splits(
    texts: Sequence[str],
    labels: Sequence[int],
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> HoldoutSplit:
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1).")
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size must be in (0, 1).")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.")

    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, y in enumerate(labels):
        by_label[int(y)].append(idx)

    rng = random.Random(seed)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for y in sorted(by_label):
        cls_idx = by_label[y][:]
        rng.shuffle(cls_idx)
        n_train, n_val, n_test = _split_counts(len(cls_idx), test_size=test_size, val_size=val_size)

        train_idx.extend(cls_idx[:n_train])
        val_idx.extend(cls_idx[n_train : n_train + n_val])
        test_idx.extend(cls_idx[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    def pick(idxs: list[int]) -> tuple[list[str], list[int]]:
        return [texts[i] for i in idxs], [int(labels[i]) for i in idxs]

    train_texts, y_train = pick(train_idx)
    val_texts, y_val = pick(val_idx)
    test_texts, y_test = pick(test_idx)

    return {
        "train_texts": train_texts,
        "val_texts": val_texts,
        "test_texts": test_texts,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }
