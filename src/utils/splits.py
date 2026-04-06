from __future__ import annotations

from typing import Sequence, TypedDict

from sklearn.model_selection import train_test_split


class HoldoutSplit(TypedDict):
    train_texts: list[str]
    val_texts: list[str]
    test_texts: list[str]
    y_train: list[int]
    y_val: list[int]
    y_test: list[int]


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

    trval_texts, test_texts, y_trval, y_test = train_test_split(
        list(texts),
        list(labels),
        stratify=list(labels),
        test_size=test_size,
        random_state=seed,
    )

    rel_val = val_size / (1.0 - test_size)
    train_texts, val_texts, y_train, y_val = train_test_split(
        trval_texts,
        y_trval,
        stratify=y_trval,
        test_size=rel_val,
        random_state=seed,
    )

    return {
        "train_texts": train_texts,
        "val_texts": val_texts,
        "test_texts": test_texts,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }
