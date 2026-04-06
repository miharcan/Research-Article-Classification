from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    import torch
except Exception:  # pragma: no cover - optional during lightweight CLI usage
    torch = None


ALLOWED_TEXT_REPRESENTATIONS = {"abstract", "triples", "abstract_triples", "hybrid"}
ALLOWED_CLUSTER_SELECTION_MODES = {"unsupervised", "label_aware"}


@dataclass
class AppConfig:
    json_path: str = "/home/miharc/work/datasets/archive/arxiv-metadata-oai-snapshot.json"
    load_n_clustering: int = 5000
    load_n_classifier: int = 10000
    n_train: int = 24
    force_k: int | None = None
    text_representation_cluster: str = "hybrid"
    text_representation_class: str = "hybrid"
    spacy_model: str = "en_core_web_sm"
    seed: int = 42
    deterministic: bool = True
    device: str = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    embedding_models: dict[str, str] = field(
        default_factory=lambda: {
            "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
            "MPNet": "sentence-transformers/all-mpnet-base-v2",
            "SciBERT": "allenai/scibert_scivocab_uncased",
            "SPECTER": "sentence-transformers/allenai-specter",
        }
    )
    classification_candidates: list[str] = field(
        default_factory=lambda: [
            "bert-base-uncased",
            "distilbert-base-uncased",
            "allenai/scibert_scivocab_uncased",
            "sentence-transformers/allenai-specter",
            "roberta-base",
        ]
    )
    cluster_methods: list[str] = field(default_factory=lambda: ["kmeans", "gmm", "hdbscan"])
    cluster_selection_mode: str = "unsupervised"


def _load_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping at root: {p}")
    return payload


def _validate(cfg: AppConfig, require_dataset: bool) -> None:
    if cfg.text_representation_cluster not in ALLOWED_TEXT_REPRESENTATIONS:
        raise ValueError(
            f"Invalid cluster text representation: {cfg.text_representation_cluster}. "
            f"Allowed={sorted(ALLOWED_TEXT_REPRESENTATIONS)}"
        )
    if cfg.text_representation_class not in ALLOWED_TEXT_REPRESENTATIONS:
        raise ValueError(
            f"Invalid class text representation: {cfg.text_representation_class}. "
            f"Allowed={sorted(ALLOWED_TEXT_REPRESENTATIONS)}"
        )
    if cfg.cluster_selection_mode not in ALLOWED_CLUSTER_SELECTION_MODES:
        raise ValueError(
            f"Invalid cluster_selection_mode: {cfg.cluster_selection_mode}. "
            f"Allowed={sorted(ALLOWED_CLUSTER_SELECTION_MODES)}"
        )

    if cfg.load_n_clustering <= 0 or cfg.load_n_classifier <= 0:
        raise ValueError("load_n_clustering/load_n_classifier must be > 0.")

    if cfg.n_train <= 0:
        raise ValueError("n_train must be > 0.")

    if cfg.force_k is not None and cfg.force_k < 2:
        raise ValueError("force_k must be >= 2 when provided.")

    if require_dataset and not Path(cfg.json_path).exists():
        raise FileNotFoundError(f"Dataset file not found: {cfg.json_path}")


def build_config(
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
    require_dataset: bool = True,
) -> AppConfig:
    cfg = AppConfig()
    payload = _load_yaml(config_path)

    merged = dict(payload)
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})

    # Env override for dataset path (useful for CI/portability).
    env_json = os.getenv("RAC_JSON_PATH")
    if env_json:
        merged["json_path"] = env_json

    # Map known keys only.
    for key, value in merged.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config key: {key}")
        setattr(cfg, key, value)

    _validate(cfg, require_dataset=require_dataset)
    return cfg


_CURRENT_CONFIG: AppConfig | None = None

# Backward-compatible module globals (used across current codebase).
JSON_PATH = ""
LOAD_N_CLUSTERING = 0
LOAD_N_CLASSIFIER = 0
n_train = 0
FORCE_K = None
EMBEDDING_MODELS: dict[str, str] = {}
CLASSIFICATION_CANDIDATES: list[str] = []
CLUSTER_METHODS: list[str] = []
CLUSTER_SELECTION_MODE = "unsupervised"
TEXT_REPRESENTATION_CLUSTER = ""
TEXT_REPRESENTATION_CLASS = ""
DEVICE = "cpu"
SEED = 42
DETERMINISTIC = True
nlp = None
_NLP_MODEL_NAME = "en_core_web_sm"


def _apply_config(cfg: AppConfig) -> AppConfig:
    global _CURRENT_CONFIG
    global JSON_PATH, LOAD_N_CLUSTERING, LOAD_N_CLASSIFIER, n_train, FORCE_K
    global EMBEDDING_MODELS, CLASSIFICATION_CANDIDATES, CLUSTER_METHODS
    global CLUSTER_SELECTION_MODE
    global TEXT_REPRESENTATION_CLUSTER, TEXT_REPRESENTATION_CLASS
    global DEVICE, SEED, DETERMINISTIC, nlp, _NLP_MODEL_NAME

    _CURRENT_CONFIG = cfg

    JSON_PATH = cfg.json_path
    LOAD_N_CLUSTERING = cfg.load_n_clustering
    LOAD_N_CLASSIFIER = cfg.load_n_classifier
    n_train = cfg.n_train
    FORCE_K = cfg.force_k

    EMBEDDING_MODELS = cfg.embedding_models
    CLASSIFICATION_CANDIDATES = cfg.classification_candidates
    CLUSTER_METHODS = cfg.cluster_methods
    CLUSTER_SELECTION_MODE = cfg.cluster_selection_mode

    TEXT_REPRESENTATION_CLUSTER = cfg.text_representation_cluster
    TEXT_REPRESENTATION_CLASS = cfg.text_representation_class
    DEVICE = cfg.device
    SEED = cfg.seed
    DETERMINISTIC = cfg.deterministic

    _NLP_MODEL_NAME = cfg.spacy_model
    nlp = None
    return cfg


def configure_runtime(
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
    require_dataset: bool = True,
) -> AppConfig:
    cfg = build_config(
        config_path=config_path,
        overrides=overrides,
        require_dataset=require_dataset,
    )
    return _apply_config(cfg)


def get_config() -> AppConfig:
    if _CURRENT_CONFIG is None:
        raise RuntimeError("Config has not been initialized yet.")
    return _CURRENT_CONFIG


def get_nlp() -> Any:
    global nlp
    if nlp is None:
        import spacy

        nlp = spacy.load(_NLP_MODEL_NAME)
    return nlp


# Initialize defaults for modules that import config outside main entrypoint.
configure_runtime(require_dataset=False)
