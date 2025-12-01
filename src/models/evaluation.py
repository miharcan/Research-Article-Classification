# -------------------------------------------------
# EVALUATION
# -------------------------------------------------
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score,
    top_k_accuracy_score)
from sklearn.preprocessing import label_binarize
from utils.logging_utils import logger

def analyze_clusters(df_cluster, df_class):
    """
    Provide research-quality analysis:
    - For each cluster: category distribution
    - Cluster purity / entropy (approx)
    """
    logger.info("=== Cluster Analysis on df_cluster ===")
    for cid, grp in df_cluster.groupby("cluster_id"):
        counts = grp["top_category"].value_counts().head(10)
        logger.info("Cluster %s (%d docs) top categories: %s",
                    cid, len(grp), counts.to_dict())

    logger.info("=== Cluster Analysis on df_class (assigned) ===")
    for cid, grp in df_class.groupby("cluster_id"):
        counts = grp["top_category"].value_counts().head(10)
        logger.info("Cluster %s (%d docs) top categories: %s",
                    cid, len(grp), counts.to_dict())
        

def evaluate_predictions_full(y_true, logits, num_labels):
    """Compute full evaluation suite from logits + true labels."""
    y_pred = logits.argmax(axis=1)

    metrics = {}

    metrics["acc"] = accuracy_score(y_true, y_pred)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")

    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics["kappa"] = cohen_kappa_score(y_true, y_pred)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    # Top-3 accuracy
    try:
        metrics["top3_acc"] = top_k_accuracy_score(y_true, logits, k=3)
    except Exception:
        metrics["top3_acc"] = None

    # Multi-class ROC-AUC
    try:
        y_bin = label_binarize(y_true, classes=list(range(num_labels)))
        metrics["roc_auc_macro_ovr"] = roc_auc_score(
            y_bin, logits, average="macro", multi_class="ovr"
        )
    except Exception:
        metrics["roc_auc_macro_ovr"] = None

    return metrics