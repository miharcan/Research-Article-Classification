from __future__ import annotations

import pandas as pd

from utils.config import TEXT_REPRESENTATION_CLASS, TEXT_REPRESENTATION_CLUSTER
from utils.logging_utils import logger


def prepare_text_representations(df: pd.DataFrame, mode: str | None = None) -> pd.DataFrame:
    """
    Ensure df has clean text representations used across clustering and classification.
    """
    if "clean" not in df:
        raise ValueError("df must contain df['clean'] before preparing representations.")

    if "triples" not in df:
        logger.warning("df['triples'] missing -> using placeholders")
        df["triples"] = [["(unk, unk, unk)"]] * len(df)

    def _triples_to_str(x: object) -> str:
        return x if isinstance(x, str) else " ; ".join(map(str, x))

    df["triples_str"] = df["triples"].apply(_triples_to_str)
    df["abstract_triples"] = df["clean"] + " " + df["triples_str"]

    if "graph_text" not in df:
        df["graph_text"] = [""] * len(df)

    df["hybrid"] = (
        "ABSTRACT: "
        + df["clean"]
        + " [SEP] TRIPLES: "
        + df["triples_str"]
        + " [SEP] GRAPH: "
        + df["graph_text"]
    )

    logger.info("Prepared text representations. Using mode: %s", mode)
    return df


def select_texts_for_clustering(df: pd.DataFrame) -> list[str]:
    """Return texts used for clustering embeddings."""
    mode = TEXT_REPRESENTATION_CLUSTER
    logger.info("[CLUSTERING] Using text representation: %s", mode)
    return _select_text_column(df, mode)


def select_texts_for_classification(df: pd.DataFrame) -> list[str]:
    """Return texts used for classification."""
    mode = TEXT_REPRESENTATION_CLASS
    logger.info("[CLASSIFICATION] Using text representation: %s", mode)
    return _select_text_column(df, mode)


def _select_text_column(df: pd.DataFrame, mode: str) -> list[str]:
    if mode == "abstract":
        return df["clean"].tolist()
    if mode == "triples":
        return df["triples"].astype(str).tolist()
    if mode == "abstract_triples":
        return df["abstract_triples"].tolist()
    if mode == "hybrid":
        if "hybrid" not in df:
            raise RuntimeError("df['hybrid'] missing - call prepare_text_representations first.")
        return df["hybrid"].tolist()
    raise ValueError(f"Unknown text representation mode: {mode}")
