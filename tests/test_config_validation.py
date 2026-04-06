from __future__ import annotations

from pathlib import Path

import pytest

from utils.config import build_config


def test_build_config_rejects_invalid_text_representation() -> None:
    with pytest.raises(ValueError):
        build_config(
            overrides={"text_representation_cluster": "invalid_mode"},
            require_dataset=False,
        )


def test_build_config_rejects_invalid_cluster_selection_mode() -> None:
    with pytest.raises(ValueError):
        build_config(
            overrides={"cluster_selection_mode": "invalid_mode"},
            require_dataset=False,
        )


def test_build_config_rejects_missing_dataset_when_required(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.jsonl"
    with pytest.raises(FileNotFoundError):
        build_config(overrides={"json_path": str(missing)}, require_dataset=True)


def test_build_config_accepts_valid_override(tmp_path: Path) -> None:
    dataset = tmp_path / "sample.jsonl"
    dataset.write_text('{"abstract":"a","categories":"cs.AI"}\n', encoding="utf-8")
    cfg = build_config(
        overrides={
            "json_path": str(dataset),
            "text_representation_cluster": "abstract",
            "text_representation_class": "hybrid",
            "cluster_selection_mode": "unsupervised",
            "seed": 7,
        },
        require_dataset=True,
    )
    assert cfg.seed == 7
    assert cfg.text_representation_cluster == "abstract"
    assert cfg.text_representation_class == "hybrid"
    assert cfg.cluster_selection_mode == "unsupervised"
