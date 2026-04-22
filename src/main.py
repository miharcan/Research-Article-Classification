from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.config import (
    ALLOWED_TEXT_REPRESENTATIONS,
    ALLOWED_CLUSTER_SELECTION_MODES,
    configure_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research Article Classification pipeline")
    parser.add_argument(
        "cluster_text_mode",
        nargs="?",
        choices=sorted(ALLOWED_TEXT_REPRESENTATIONS),
        default=None,
        help="Text representation for clustering stage.",
    )
    parser.add_argument(
        "class_text_mode",
        nargs="?",
        choices=sorted(ALLOWED_TEXT_REPRESENTATIONS),
        default=None,
        help="Text representation for classification stage.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config file path.",
    )
    parser.add_argument("--json-path", default=None, help="Path to arXiv JSONL dataset.")
    parser.add_argument("--load-n-clustering", type=int, default=None)
    parser.add_argument("--load-n-classifier", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None, help="Optuna trial count.")
    parser.add_argument("--force-k", type=int, default=None, help="Force K for KMeans/GMM.")
    parser.add_argument(
        "--cluster-selection-mode",
        choices=sorted(ALLOWED_CLUSTER_SELECTION_MODES),
        default=None,
        help="unsupervised (label-free) or label_aware pipeline selection.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--export-json",
        default=None,
        help="Optional path to export run summary JSON (for benchmark aggregation).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import mlflow

    overrides = {
        "json_path": args.json_path,
        "load_n_clustering": args.load_n_clustering,
        "load_n_classifier": args.load_n_classifier,
        "n_train": args.n_trials,
        "force_k": args.force_k,
        "cluster_selection_mode": args.cluster_selection_mode,
        "seed": args.seed,
        "text_representation_cluster": args.cluster_text_mode,
        "text_representation_class": args.class_text_mode,
    }
    cfg = configure_runtime(config_path=args.config, overrides=overrides, require_dataset=True)

    # Import after config init so modules see finalized runtime values.
    from utils.reproducibility import set_global_seed
    from data.load_data import prepare_datasets
    from models.clustering import (
        compare_embeddings_and_clusterers,
        select_best_pipeline,
        fit_final_clusterer,
        assign_clusters_to_class_set,
    )
    from sklearn.preprocessing import LabelEncoder
    from models.tuning import run_hyperparameter_search
    from models.evaluation import analyze_clusters
    from utils.logging_utils import logger, rebind_file_handler, log_path, log_protocol_metadata
    from models.text_selection import prepare_text_representations

    set_global_seed(cfg.seed, cfg.deterministic)
    logger.info("Starting pipeline with config: %s", cfg)
    logger.info("Reproducibility seed=%d deterministic=%s", cfg.seed, cfg.deterministic)
    log_protocol_metadata(cfg)

    # 1) Data
    df_cluster, df_class = prepare_datasets()
    df_cluster = prepare_text_representations(df_cluster, mode=cfg.text_representation_cluster)
    df_class = prepare_text_representations(df_class, mode=cfg.text_representation_class)

    # 2) Embedding × clustering comparison
    compare_df, best_k_km, best_k_gmm = compare_embeddings_and_clusterers(df_cluster)

    # 3) Select best pipeline
    best_pipeline = select_best_pipeline(compare_df, n_samples=len(df_cluster))

    # 4) Fit final clusterer on df_cluster
    clusterer = fit_final_clusterer(df_cluster, best_pipeline)

    # 5) Assign clusters to df_class using correct prediction logic
    assign_clusters_to_class_set(df_class, best_pipeline, clusterer)

    # 6) Research-quality cluster analysis
    analyze_clusters(df_cluster, df_class)

    # 6b) DROP clusters with too few samples (e.g. < 2) from df_class
    vc = df_class["cluster_id"].value_counts()
    valid_clusters = vc[vc >= 2].index
    dropped = len(df_class) - len(df_class[df_class["cluster_id"].isin(valid_clusters)])
    logger.info("Dropping %d samples from rare clusters (size < 2)", dropped)
    df_class = df_class[df_class["cluster_id"].isin(valid_clusters)].reset_index(drop=True)

    # 7) Encode labels ONCE for Optuna
    le = LabelEncoder()
    df_class["cluster_id_enc"] = le.fit_transform(df_class["cluster_id"])

    # IMPORTANT: FIX LOGGING BREAKAGE CAUSED BY MLFLOW / OPTUNA
    mlflow.set_experiment("ArXiv_Classifier_Optimisation")
    rebind_file_handler(log_path)

    # 8) Run Optuna search
    study = run_hyperparameter_search(df_class)
    logger.info("Best hyperparameters: %s", study.best_trial.params)
    logger.info("Done. Log file: %s", log_path)

    if args.export_json:
        export_path = Path(args.export_json)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "config": {
                "json_path": cfg.json_path,
                "load_n_clustering": cfg.load_n_clustering,
                "load_n_classifier": cfg.load_n_classifier,
                "n_train": cfg.n_train,
                "force_k": cfg.force_k,
                "text_representation_cluster": cfg.text_representation_cluster,
                "text_representation_class": cfg.text_representation_class,
                "cluster_selection_mode": cfg.cluster_selection_mode,
                "seed": cfg.seed,
                "deterministic": cfg.deterministic,
                "spacy_model": cfg.spacy_model,
                "holdout_test_size": cfg.holdout_test_size,
                "holdout_val_size": cfg.holdout_val_size,
                "early_stopping_patience": cfg.early_stopping_patience,
                "early_stopping_min_delta": cfg.early_stopping_min_delta,
                "device": cfg.device,
                "embedding_models": cfg.embedding_models,
                "classification_candidates": cfg.classification_candidates,
                "cluster_methods": cfg.cluster_methods,
            },
            "best_k_kmeans": best_k_km,
            "best_k_gmm": best_k_gmm,
            "best_pipeline": best_pipeline,
            "clustering_comparison": compare_df.to_dict(orient="records"),
            "best_trial": {
                "number": study.best_trial.number,
                "value": study.best_value,
                "params": study.best_trial.params,
            },
            "holdout_test_metrics": study.user_attrs.get("holdout_test_metrics", {}),
            "log_path": str(log_path),
        }
        with export_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Exported run summary JSON: %s", export_path)


if __name__ == "__main__":
    main()
