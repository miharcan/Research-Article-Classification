# -------------------------------------------------
# DATA LOADING & PREPROCESSING
# -------------------------------------------------
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from data.preprocess import extract_triples
from utils.config import (
    JSON_PATH,
    LOAD_N_CLASSIFIER,
    LOAD_N_CLUSTERING,
    TEXT_REPRESENTATION_CLASS,
    TEXT_REPRESENTATION_CLUSTER,
)
from utils.logging_utils import logger


def _select_text_column(df: pd.DataFrame, mode: str) -> list[str] | None:
    """Internal helper — picks correct column based on mode."""
    logger.info("Using text representation: %s", mode)

    if mode == "abstract":
        return df["clean"].tolist()

    if mode == "triples":
        return df["triples"].apply(
            lambda x: " ; ".join([f"{s} {r} {o}" for (s, r, o) in x])
            if isinstance(x, list)
            else str(x)
        ).tolist()

    if mode == "abstract_triples":
        return df["abstract_triples"].tolist()

    if mode == "hybrid":
        # hybrid is composed in embeddings.py
        return None

    raise ValueError(f"Unknown text representation mode: {mode}")


def select_cluster_texts(df: pd.DataFrame) -> list[str] | None:
    """Text for clustering embeddings."""
    return _select_text_column(df, TEXT_REPRESENTATION_CLUSTER)


def select_class_texts(df: pd.DataFrame) -> list[str] | None:
    """Text for classification / BERT fine-tuning."""
    return _select_text_column(df, TEXT_REPRESENTATION_CLASS)


def load_json_subset(path: str, limit: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            try:
                obj = json.loads(line)
                if "abstract" in obj and "categories" in obj:
                    rows.append(obj)
            except Exception:
                continue
    return pd.DataFrame(rows)


def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", str(t).lower()).strip()


def top_cat_from_categories(cat_str: str) -> str:
    """
    Take the first arxiv category and split at '.', e.g. 'hep-th' -> 'hep-th', 'cs.LG' -> 'cs'.
    """
    if not isinstance(cat_str, str) or not cat_str.strip():
        return "unknown"
    parts = cat_str.split()
    first = parts[0]
    return first.split(".")[0]


def build_augmented_text(
    abstract: str,
    triples: str | list[tuple[str, str, str]],
    nodes: list[str] | None = None,
    edges: list[tuple[str, str]] | None = None,
) -> str:
    """
    Build combined representation from abstract + graph/knowledge snippets.
    """
    text_blocks = [f"ABSTRACT:\n{abstract.strip()}"]

    if isinstance(triples, str):
        triple_str = triples
    else:
        triple_str = " ; ".join([f"{s} {r} {o}" for (s, r, o) in triples])
    text_blocks.append(f"KNOWLEDGE TRIPLES:\n{triple_str}")

    if nodes:
        node_str = ", ".join(nodes)
        text_blocks.append(f"GRAPH NODES:\n{node_str}")

    if edges:
        edge_str = " ; ".join([f"{src} -> {dst}" for (src, dst) in edges])
        text_blocks.append(f"GRAPH EDGES:\n{edge_str}")

    return "\n\n".join(text_blocks)


def prepare_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a random subset of the data, then split into:
    - df_cluster: for unsupervised clustering
    - df_class:   for classifier training
    """
    total_needed = LOAD_N_CLUSTERING + LOAD_N_CLASSIFIER
    df_all = load_json_subset(JSON_PATH, total_needed)
    logger.info("Loaded %d total rows", len(df_all))

    df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_all["clean"] = df_all["abstract"].astype(str).apply(clean_text)
    df_all["triples"] = df_all["abstract"].astype(str).apply(extract_triples)
    df_all["abstract_triples"] = df_all.apply(
        lambda row: build_augmented_text(
            abstract=row["abstract"],
            triples=row["triples"],
            nodes=row.get("nodes", None),
            edges=row.get("edges", None),
        ),
        axis=1,
    )
    df_all["top_category"] = df_all["categories"].astype(str).apply(top_cat_from_categories)

    df_cluster = df_all.iloc[:LOAD_N_CLUSTERING].reset_index(drop=True)
    df_class = df_all.iloc[LOAD_N_CLUSTERING : LOAD_N_CLUSTERING + LOAD_N_CLASSIFIER].reset_index(drop=True)

    logger.info("df_cluster: %d rows, df_class: %d rows", len(df_cluster), len(df_class))
    logger.info("Top categories in df_cluster: %s", df_cluster["top_category"].value_counts().head().to_dict())

    return df_cluster, df_class
