# -------------------------------------------------
# DATA LOADING & PREPROCESSING
# -------------------------------------------------

from utils.config import (
    JSON_PATH,
    LOAD_N_CLUSTERING,
    LOAD_N_CLASSIFIER
)
import json
import re
import pandas as pd
from utils.logging_utils import logger

def load_json_subset(path, limit):
    rows = []
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
    df = pd.DataFrame(rows)
    return df

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
    return first.split(".")[0]  # first.split(".")[0] to get cs.AI -> cs

def prepare_datasets():
    """
    Load a random subset of the data, then split into:
    - df_cluster: for unsupervised clustering
    - df_class:   for classifier training
    This avoids positional / chronological bias.
    """
    total_needed = LOAD_N_CLUSTERING + LOAD_N_CLASSIFIER
    df_all = load_json_subset(JSON_PATH, total_needed)
    logger.info("Loaded %d total rows", len(df_all))

    # Shuffle once to remove chronological bias
    df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)

    df_all["clean"] = df_all["abstract"].astype(str).apply(clean_text)
    df_all["top_category"] = df_all["categories"].astype(str).apply(top_cat_from_categories)

    df_cluster = df_all.iloc[:LOAD_N_CLUSTERING].reset_index(drop=True)
    df_class   = df_all.iloc[LOAD_N_CLUSTERING:LOAD_N_CLUSTERING + LOAD_N_CLASSIFIER].reset_index(drop=True)

    logger.info("df_cluster: %d rows, df_class: %d rows", len(df_cluster), len(df_class))
    logger.info("Top categories in df_cluster: %s", df_cluster["top_category"].value_counts().head().to_dict())

    return df_cluster, df_class