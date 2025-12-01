import mlflow

from utils.config import *
from data.load_data import prepare_datasets
from models.embeddings import get_embeddings
from models.clustering import (
    compare_embeddings_and_clusterers,
    select_best_pipeline,
    fit_final_clusterer,
    assign_clusters_to_class_set,
)
from sklearn.preprocessing import LabelEncoder
from models.tuning import run_hyperparameter_search
from models.evaluation import analyze_clusters
from utils.logging_utils import logger, rebind_file_handler, log_path

if __name__ == "__main__":
    logger.info("Starting pipeline...")

    # 1) Data
    df_cluster, df_class = prepare_datasets()

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

    # 7) Encode labels ONCE for Optuna
    le = LabelEncoder()
    df_class["cluster_id_enc"] = le.fit_transform(df_class["cluster_id"])

    # ---------------------------------------------------------
    # IMPORTANT: FIX LOGGING BREAKAGE CAUSED BY MLFLOW / OPTUNA
    # ---------------------------------------------------------
    mlflow.set_experiment("ArXiv_Classifier_Optimisation")
    rebind_file_handler(log_path) 

    # 8) Run Optuna search
    # study = run_hyperparameter_search()
    study = run_hyperparameter_search(df_class)
    logger.info("Best hyperparameters: %s", study.best_trial.params)

    logger.info("Done. Log file: %s", log_path)