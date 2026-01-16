from utils.logging_utils import logger
from utils.config import (
    TEXT_REPRESENTATION_CLUSTER,
    TEXT_REPRESENTATION_CLASS
)

def prepare_text_representations(df, mode=None):
    """
    Ensures df contains:
        df["clean"]
        df["triples"]
        df["abstract_triples"]
        df["hybrid"]
    
    Any missing representation is generated or replaced with a fallback string.
    """

    if "clean" not in df:
        raise ValueError("df must contain df['clean'] before preparing representations.")

    # 1) Triples fallback
    if "triples" not in df:
        logger.warning("df['triples'] missing → using ['(unk,unk,unk)'] placeholders")
        df["triples"] = [["(unk, unk, unk)"]] * len(df)

    # Convert triples to string
    def _triples_to_str(x):
        return x if isinstance(x, str) else " ; ".join(map(str, x))

    df["triples_str"] = df["triples"].apply(_triples_to_str)

    # 2) abstract_triples
    df["abstract_triples"] = df["clean"] + " " + df["triples_str"]

    # 3) hybrid mode requires three components
    if "graph_text" not in df:
        df["graph_text"] = [""] * len(df)

    df["hybrid"] = (
        "ABSTRACT: " + df["clean"] 
        + " [SEP] TRIPLES: " + df["triples_str"] 
        + " [SEP] GRAPH: " + df["graph_text"]
    )


    logger.info(f"Prepared text representations. Using mode: {mode}")
    return df


# -------------------------------------------------
# TEXT REPRESENTATION SELECTION (CLUSTERING + CLASSIFICATION)
# -------------------------------------------------

from utils.config import (
    TEXT_REPRESENTATION_CLUSTER,
    TEXT_REPRESENTATION_CLASS
)
from utils.logging_utils import logger


def select_texts_for_clustering(df):
    """Return texts used for clustering embeddings."""
    mode = TEXT_REPRESENTATION_CLUSTER
    logger.info(f"[CLUSTERING] Using text representation: {mode}")

    return _select_text_column(df, mode)


def select_texts_for_classification(df):
    """Return texts used for classification (BERT fine-tuning)."""
    mode = TEXT_REPRESENTATION_CLASS
    logger.info(f"[CLASSIFICATION] Using text representation: {mode}")

    return _select_text_column(df, mode)


def _select_text_column(df, mode: str):
    """Internal selector — returns a list of strings based on the mode."""

    if mode == "abstract":
        return df["clean"].tolist()


    elif mode == "triples":
    # triples are already stored as scibert_friendly_text → return as-is
        return df["triples"].astype(str).tolist()

    
    elif mode == "abstract_triples":
        return df["abstract_triples"].tolist()

    elif mode == "hybrid":
        if "hybrid" not in df:
            raise RuntimeError("df['hybrid'] missing — call prepare_text_representations first.")
        return df["hybrid"].tolist()

    else:
        raise ValueError(f"Unknown text representation mode: {mode}")
