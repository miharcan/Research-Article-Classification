from __future__ import annotations

from collections import Counter

import pytest

from utils.splits import prepare_holdout_splits


def _make_balanced_texts_and_labels(n_per_class: int = 50) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    for label in (0, 1):
        for i in range(n_per_class):
            texts.append(f"text_{label}_{i}")
            labels.append(label)
    return texts, labels


def test_prepare_holdout_splits_disjoint_and_complete() -> None:
    texts, labels = _make_balanced_texts_and_labels(n_per_class=50)
    out = prepare_holdout_splits(texts, labels, seed=42, test_size=0.2, val_size=0.2)

    train_set = set(out["train_texts"])
    val_set = set(out["val_texts"])
    test_set = set(out["test_texts"])

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert len(train_set | val_set | test_set) == len(texts)


def test_prepare_holdout_splits_is_stratified() -> None:
    texts, labels = _make_balanced_texts_and_labels(n_per_class=50)
    out = prepare_holdout_splits(texts, labels, seed=42, test_size=0.2, val_size=0.2)

    assert Counter(out["y_train"]) == Counter({0: 30, 1: 30})
    assert Counter(out["y_val"]) == Counter({0: 10, 1: 10})
    assert Counter(out["y_test"]) == Counter({0: 10, 1: 10})


def test_prepare_holdout_splits_rejects_invalid_sizes() -> None:
    texts, labels = _make_balanced_texts_and_labels(n_per_class=10)
    with pytest.raises(ValueError):
        prepare_holdout_splits(texts, labels, seed=1, test_size=0.6, val_size=0.5)
