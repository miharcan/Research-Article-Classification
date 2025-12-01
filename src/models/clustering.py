# -------------------------------------------------
# CLUSTERING HELPERS
# -------------------------------------------------
import numpy as np
import pandas as pd
from models.embeddings import get_embeddings
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, confusion_matrix, adjusted_rand_score, normalized_mutual_info_score, classification_report,
    accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score,
    top_k_accuracy_score)
from sklearn.mixture import GaussianMixture
import hdbscan
from hdbscan import approximate_predict
from utils.logging_utils import logger
from utils.config import *


def best_k_sweep(X, top_categories, k_range):
    """
    Find best K for KMeans / GMM based on silhouette + NMI/ARI.
    """
    best = {
        "kmeans": {"k": None, "score": -np.inf},
        "gmm":    {"k": None, "score": -np.inf},
    }

    y = top_categories

    for k in k_range:
        # KMeans
        try:
            km_labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
            sil = silhouette_score(X, km_labels)
            ari = adjusted_rand_score(y, km_labels)
            nmi = normalized_mutual_info_score(y, km_labels)
            # composite
            score = nmi + 0.5 * ari + 0.5 * sil
            if score > best["kmeans"]["score"]:
                best["kmeans"] = {"k": k, "score": score}
        except Exception as e:
            logger.warning("KMeans k=%d failed: %s", k, str(e))

        # GMM
        try:
            gmm_labels = GaussianMixture(
                n_components=k, random_state=42, reg_covar=1e-5
            ).fit_predict(X)
            sil = silhouette_score(X, gmm_labels)
            ari = adjusted_rand_score(y, gmm_labels)
            nmi = normalized_mutual_info_score(y, gmm_labels)
            score = nmi + 0.5 * ari + 0.5 * sil
            if score > best["gmm"]["score"]:
                best["gmm"] = {"k": k, "score": score}
        except Exception as e:
            logger.warning("GMM k=%d failed: %s", k, str(e))

    return best["kmeans"]["k"], best["gmm"]["k"]


def run_kmeans(X, y, k):
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
    return dict(
        algorithm="KMeans",
        k=k,
        clusters=len(set(labels)),
        noise=0,
        ari=adjusted_rand_score(y, labels),
        nmi=normalized_mutual_info_score(y, labels),
        silhouette=silhouette_score(X, labels),
    )


def run_gmm(X, y, k):
    mdl = GaussianMixture(n_components=k, random_state=42, reg_covar=1e-5)
    labels = mdl.fit_predict(X)
    return dict(
        algorithm="GMM",
        k=k,
        clusters=len(set(labels)),
        noise=0,
        ari=adjusted_rand_score(y, labels),
        nmi=normalized_mutual_info_score(y, labels),
        silhouette=silhouette_score(X, labels),
    )


def run_hdbscan(X, y):
    """
    Sweep min_cluster_size and pick best based on:
      composite_score = nmi + 0.5*ari - noise_penalty
    We treat HDBSCAN explicitly differently because silhouette is not well-defined with noise.
    """
    best_row = None
    for m in [5, 10, 20, 30, 50, 75]:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=m)
        labels = clusterer.fit_predict(X)
        noise = int(np.sum(labels == -1))
        if len(set(labels)) <= 1:
            continue
        safe = np.where(labels == -1, labels.max() + 1, labels)
        ari = adjusted_rand_score(y, safe)
        nmi = normalized_mutual_info_score(y, safe)
        noise_frac = noise / len(labels)

        # composite score: reward NMI/ARI, penalize noise
        score = nmi + 0.5 * ari - 0.5 * noise_frac

        row = dict(
            algorithm="HDBSCAN",
            min_cluster_size=m,
            clusters=len(set(safe)),
            noise=noise,
            noise_frac=noise_frac,
            ari=ari,
            nmi=nmi,
            silhouette=None,
            composite=score,
        )

        if best_row is None or row["composite"] > best_row["composite"]:
            best_row = row

    return best_row


def compare_embeddings_and_clusterers(df_cluster):
    """
    For each embedding model, run KMeans, GMM, HDBSCAN (with tuned K/min_cluster_size)
    and collect ARI, NMI, silhouette, noise, etc.
    """
    texts = df_cluster["clean"].tolist()
    y     = df_cluster["top_category"].tolist()

    # base MiniLM embeddings only for K sweep
    X_base = get_embeddings(texts, "MiniLM", subset_id="cluster")
    k_range = range(2, min(df_cluster["top_category"].nunique(), 40) + 1, 2)
    best_k_km, best_k_gmm = best_k_sweep(X_base, y, k_range)

    logger.info("Best K for KMeans: %s, GMM: %s", best_k_km, best_k_gmm)

    rows = []
    for emb_name in EMBEDDING_MODELS.keys():
        logger.info("==== Embedding: %s ====", emb_name)
        X = get_embeddings(texts, emb_name, subset_id="cluster")

        if "kmeans" in CLUSTER_METHODS and best_k_km is not None:
            rows.append({**run_kmeans(X, y, best_k_km), "embedding": emb_name})
        if "gmm" in CLUSTER_METHODS and best_k_gmm is not None:
            rows.append({**run_gmm(X, y, best_k_gmm), "embedding": emb_name})
        if "hdbscan" in CLUSTER_METHODS:
            hdb_row = run_hdbscan(X, y)
            if hdb_row is not None:
                hdb_row["embedding"] = emb_name
                rows.append(hdb_row)

    df_results = pd.DataFrame(rows)
    logger.info("Clustering comparison:\n%s", df_results.to_string(index=False))
    return df_results, best_k_km, best_k_gmm


def select_best_pipeline(df_results, n_samples):
    """
    Combine intrinsic and extrinsic metrics into a single research-quality
    selection criterion.

    Score = NMI + 0.5*ARI + 0.5*silhouette (for KMeans/GMM)
    For HDBSCAN: use 'composite' already computed.
    """
    rows = []
    for _, row in df_results.iterrows():
        if row["ari"] <= 0 or row["nmi"] <= 0:
            continue

        if row["algorithm"] in ["KMeans", "GMM"]:
            sil = row["silhouette"]
            score = row["nmi"] + 0.5 * row["ari"] + 0.5 * (sil if pd.notna(sil) else 0.0)
        else:  # HDBSCAN
            score = row.get("composite", row["nmi"] + 0.5 * row["ari"] - 0.5 * (row["noise"] / n_samples))

        rows.append({**row.to_dict(), "score": score})

    if not rows:
        raise RuntimeError("No valid clustering pipeline found.")

    df_scored = pd.DataFrame(rows).sort_values("score", ascending=False)
    logger.info("Scored pipelines:\n%s", df_scored.to_string(index=False))
    best = df_scored.iloc[0].to_dict()
    logger.info("Selected best pipeline: %s", best)

    return best

# -------------------------------------------------
# CLUSTER ASSIGNMENT
# -------------------------------------------------
def fit_final_clusterer(df_cluster, best_pipeline):
    texts = df_cluster["clean"].tolist()
    X = get_embeddings(texts, best_pipeline["embedding"], subset_id="cluster")

    alg = best_pipeline["algorithm"]
    if alg == "KMeans":
        k = int(best_pipeline["k"])
        model = KMeans(n_clusters=k, random_state=42).fit(X)
        labels = model.labels_
    elif alg == "GMM":
        k = int(best_pipeline["k"])
        model = GaussianMixture(n_components=k, random_state=42, reg_covar=1e-5).fit(X)
        labels = model.predict(X)
    else:  # HDBSCAN
        mcs = int(best_pipeline["min_cluster_size"])
        # IMPORTANT: prediction_data=True for approximate_predict
        model = hdbscan.HDBSCAN(min_cluster_size=mcs, prediction_data=True).fit(X)
        labels = model.labels_
        labels = np.where(labels == -1, labels.max() + 1, labels)

    df_cluster["cluster_id"] = labels
    logger.info("Cluster distribution on df_cluster: %s", df_cluster["cluster_id"].value_counts().to_dict())
    return model


def assign_clusters_to_class_set(df_class, best_pipeline, clusterer):
    texts = df_class["clean"].tolist()
    X = get_embeddings(texts, best_pipeline["embedding"], subset_id="class")

    alg = best_pipeline["algorithm"]
    if alg == "HDBSCAN":
        labels, strengths = approximate_predict(clusterer, X)
        labels = np.where(labels == -1, labels.max() + 1, labels)
    else:
        labels = clusterer.predict(X)

    df_class["cluster_id"] = labels
    logger.info("Cluster distribution on df_class: %s", df_class["cluster_id"].value_counts().to_dict())
