# -------------------------------------------------
# CLUSTERING HELPERS
# -------------------------------------------------
import numpy as np
import pandas as pd
from models.embeddings import get_embeddings
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.mixture import GaussianMixture
import hdbscan
from hdbscan import approximate_predict
from utils.logging_utils import logger
from utils.config import (
    CLUSTER_METHODS,
    CLUSTER_SELECTION_MODE,
    EMBEDDING_MODELS,
    FORCE_K,
)
from data.load_data import select_cluster_texts


def best_k_sweep(X, top_categories, k_range):
    """
    Find best K for KMeans / GMM based on silhouette + NMI/ARI.
    """
    best = {
        "kmeans": {"k": None, "score": -np.inf},
        "gmm":    {"k": None, "score": -np.inf},
    }

    y = top_categories
    label_aware = CLUSTER_SELECTION_MODE == "label_aware"

    for k in k_range:
        # KMeans
        try:
            km_labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
            sil = silhouette_score(X, km_labels)
            if label_aware:
                ari = adjusted_rand_score(y, km_labels)
                nmi = normalized_mutual_info_score(y, km_labels)
                score = nmi + 0.5 * ari + 0.5 * sil
            else:
                ch = calinski_harabasz_score(X, km_labels)
                db = davies_bouldin_score(X, km_labels)
                score = sil + 0.01 * np.log1p(ch) - 0.05 * db
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
            if label_aware:
                ari = adjusted_rand_score(y, gmm_labels)
                nmi = normalized_mutual_info_score(y, gmm_labels)
                score = nmi + 0.5 * ari + 0.5 * sil
            else:
                ch = calinski_harabasz_score(X, gmm_labels)
                db = davies_bouldin_score(X, gmm_labels)
                score = sil + 0.01 * np.log1p(ch) - 0.05 * db
            if score > best["gmm"]["score"]:
                best["gmm"] = {"k": k, "score": score}
        except Exception as e:
            logger.warning("GMM k=%d failed: %s", k, str(e))

    return best["kmeans"]["k"], best["gmm"]["k"]


def run_kmeans(X, y, k):
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
    ari = adjusted_rand_score(y, labels) if y is not None else None
    nmi = normalized_mutual_info_score(y, labels) if y is not None else None
    return dict(
        algorithm="KMeans",
        k=k,
        clusters=len(set(labels)),
        noise=0,
        ari=ari,
        nmi=nmi,
        silhouette=silhouette_score(X, labels),
    )


def run_gmm(X, y, k):
    mdl = GaussianMixture(n_components=k, random_state=42, reg_covar=1e-5)
    labels = mdl.fit_predict(X)
    ari = adjusted_rand_score(y, labels) if y is not None else None
    nmi = normalized_mutual_info_score(y, labels) if y is not None else None
    return dict(
        algorithm="GMM",
        k=k,
        clusters=len(set(labels)),
        noise=0,
        ari=ari,
        nmi=nmi,
        silhouette=silhouette_score(X, labels),
    )


def run_hdbscan(X, y):
    """
    Sweep min_cluster_size and pick best based on:
      composite_score = nmi + 0.5*ari - noise_penalty
    HDBSCAN explicitly differently because silhouette is not well-defined with noise.
    """
    best_row = None
    for m in [5, 10, 20, 30, 50, 75]:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=m)
        labels = clusterer.fit_predict(X)
        noise = int(np.sum(labels == -1))
        if len(set(labels)) <= 1:
            continue
        safe = np.where(labels == -1, labels.max() + 1, labels)
        noise_frac = noise / len(labels)
        sil = silhouette_score(X, safe) if len(set(safe)) > 1 else None

        # Always compute ARI/NMI for reporting if labels are available.
        # Selection objective still depends on CLUSTER_SELECTION_MODE.
        if y is not None:
            ari = adjusted_rand_score(y, safe)
            nmi = normalized_mutual_info_score(y, safe)
        else:
            ari = None
            nmi = None

        if CLUSTER_SELECTION_MODE == "label_aware" and y is not None:
            # composite score: reward NMI/ARI, penalize noise
            score = nmi + 0.5 * ari - 0.5 * noise_frac
        else:
            score = (sil if sil is not None else -1.0) - 0.5 * noise_frac

        row = dict(
            algorithm="HDBSCAN",
            min_cluster_size=m,
            clusters=len(set(safe)),
            noise=noise,
            noise_frac=noise_frac,
            ari=ari,
            nmi=nmi,
            silhouette=sil,
            composite=score,
        )

        if best_row is None or row["composite"] > best_row["composite"]:
            best_row = row

    return best_row

def compare_embeddings_and_clusterers(df_cluster):
    """
    For each embedding model:
      - compute embeddings (abstract/triples/hybrid, depending on config)
      - run KMeans, GMM, and HDBSCAN
      - collect metrics
    """

    texts = select_cluster_texts(df_cluster)

    label_aware = CLUSTER_SELECTION_MODE == "label_aware"
    # Keep labels available for ARI/NMI reporting in all modes.
    # Selection logic remains controlled by CLUSTER_SELECTION_MODE.
    y = df_cluster["top_category"].tolist()
    logger.info("Cluster selection mode: %s", CLUSTER_SELECTION_MODE)

    if FORCE_K is not None:
        best_k_km = best_k_gmm = FORCE_K
        
        logger.info(f"FORCE_K is set → Using K={FORCE_K} for all clusterers.")
    else:
        # ----------------------------
        # 2. Automatic K selection
        # ----------------------------
        X_base = get_embeddings(df_cluster, "MiniLM", subset_id="cluster")
        k_upper = min(df_cluster["top_category"].nunique(), 40) if label_aware else 40
        k_range = range(2, k_upper + 1, 2)
        best_k_km, best_k_gmm = best_k_sweep(X_base, y, k_range)
        logger.info("Best K for KMeans: %s, GMM: %s", best_k_km, best_k_gmm)

    rows = []

    # -------- Step 2: Evaluate each embedding model --------
    for emb_name in EMBEDDING_MODELS.keys():

        logger.info(f"==== Embedding: {emb_name} ====")

        X = get_embeddings(df_cluster, emb_name, subset_id="cluster")

        # KMeans
        if "kmeans" in CLUSTER_METHODS and best_k_km is not None:
            row = run_kmeans(X, y, best_k_km)
            row["embedding"] = emb_name
            rows.append(row)

        # GMM
        if "gmm" in CLUSTER_METHODS and best_k_gmm is not None:
            row = run_gmm(X, y, best_k_gmm)
            row["embedding"] = emb_name
            rows.append(row)

        # HDBSCAN
        if "hdbscan" in CLUSTER_METHODS:
            hdb_row = run_hdbscan(X, y)
            if hdb_row is not None:
                hdb_row["embedding"] = emb_name
                rows.append(hdb_row)

    df_results = pd.DataFrame(rows)
    logger.info("\nClustering comparison:\n%s", df_results.to_string(index=False))

    return df_results, best_k_km, best_k_gmm



def select_best_pipeline(df_results, n_samples):
    """
    Combine intrinsic and extrinsic metrics into a single research-quality
    selection criterion.

    Score = NMI + 0.5*ARI + 0.5*silhouette (for KMeans/GMM)
    For HDBSCAN: use 'composite' already computed.
    """
    if FORCE_K is not None:
        df_results = df_results[df_results["algorithm"] != "HDBSCAN"]
        logger.info("FORCE_K set → Excluding HDBSCAN from selection (it has no fixed K).")


    label_aware = CLUSTER_SELECTION_MODE == "label_aware"
    rows = []
    for _, row in df_results.iterrows():
        if label_aware:
            if row["ari"] is None or row["nmi"] is None:
                continue
            if row["ari"] <= 0 or row["nmi"] <= 0:
                continue

        if row["algorithm"] in ["KMeans", "GMM"]:
            sil = row["silhouette"]
            if label_aware:
                score = row["nmi"] + 0.5 * row["ari"] + 0.5 * (sil if pd.notna(sil) else 0.0)
            else:
                score = sil if pd.notna(sil) else -np.inf
        else:  # HDBSCAN
            if label_aware:
                score = row.get(
                    "composite",
                    row["nmi"] + 0.5 * row["ari"] - 0.5 * (row["noise"] / n_samples),
                )
            else:
                score = row.get("composite", -np.inf)

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
    """
    Fit the final clusterer on df_cluster using the SAME text representation
    that was used during the embedding sweep.
    """
    if FORCE_K is not None:
        logger.info(f"Using manually forced K={FORCE_K} for clusterer training.")

    # 1. Get the text representation used for clustering (abstract/triples/hybrid)
    from models.text_selection import select_texts_for_clustering
    texts = select_texts_for_clustering(df_cluster)

    # 2. Produce embeddings using CLUSTER mode
    X = get_embeddings(
        df_cluster,
        best_pipeline["embedding"],
        subset_id="cluster_fit",
        texts_override=texts
    )

    # 3. Fit clusterer
    alg = best_pipeline["algorithm"]

    if alg == "KMeans":
        k = int(best_pipeline["k"])
        model = KMeans(n_clusters=k, random_state=42).fit(X)
        labels = model.labels_

    elif alg == "GMM":
        k = int(best_pipeline["k"])
        model = GaussianMixture(
            n_components=k,
            random_state=42,
            reg_covar=1e-5
        ).fit(X)
        labels = model.predict(X)

    else:  # HDBSCAN
        mcs = int(best_pipeline["min_cluster_size"])
        model = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            prediction_data=True
        ).fit(X)
        labels = model.labels_
        labels = np.where(labels == -1, labels.max() + 1, labels)

    df_cluster["cluster_id"] = labels
    logger.info(
        "Cluster distribution on df_cluster: %s",
        df_cluster["cluster_id"].value_counts().to_dict()
    )

    return model


def assign_clusters_to_class_set(df_class, best_pipeline, clusterer):
    """
    Embed df_class using the SAME text mode and SAME embedding model used for clustering.
    """

    # 1. Get texts using *CLUSTERING* representation, not classification.
    from models.text_selection import select_texts_for_clustering
    texts = select_texts_for_clustering(df_class)

    # 2. Produce embeddings exactly matching clustering logic.
    X = get_embeddings(
        df_class,
        best_pipeline["embedding"],
        subset_id="cluster_assign",   # name doesn't matter, but must NOT be "class"
        texts_override=texts          # force use of CLUSTERING text list
    )

    # 3. Predict cluster labels
    alg = best_pipeline["algorithm"]

    if alg == "HDBSCAN":
        labels, strengths = approximate_predict(clusterer, X)
        labels = np.where(labels == -1, labels.max() + 1, labels)
    else:
        labels = clusterer.predict(X)

    df_class["cluster_id"] = labels
    logger.info("Cluster distribution on df_class: %s", df_class["cluster_id"].value_counts().to_dict())
