import torch
import optuna
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch import optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from models.classifier import ClusterDataset
from models.evaluation import evaluate_predictions_full
from utils.logging_utils import logger, rebind_file_handler, log_path
from utils.config import DEVICE, CLASSIFICATION_CANDIDATES


# =============================================================
# TRAIN + EVAL OF ONE MODEL (used by Optuna)
# =============================================================
def train_eval_single(
    model_name,
    lr,
    batch_size,
    epochs,
    train_texts,
    val_texts,
    y_train,
    y_val,
    num_labels,
):
    """Train one classifier config and return all logits + labels."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = ClusterDataset(train_texts, y_train, tokenizer)
    val_ds   = ClusterDataset(val_texts,   y_val, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # ---------------- TRAIN ----------------
    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE)
            )
            out.loss.backward()
            optimizer.step()

    # ---------------- EVAL ----------------
    model.eval()
    preds, trues = [], []
    all_logits = []
    
    with torch.no_grad():
        for batch in val_loader:
            logits = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE)
            ).logits

            all_logits.append(logits.cpu().numpy())
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            trues.extend(batch["labels"].numpy())

    logits_full = np.vstack(all_logits)

    return np.array(trues), logits_full, model


# =============================================================
# OPTUNA OBJECTIVE
# =============================================================
def objective(trial, df_class):
    """Wrapped objective receives df_class from outside."""

    model_name = trial.suggest_categorical("model_name", CLASSIFICATION_CANDIDATES)
    lr         = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch", [8, 16, 32])
    epochs     = trial.suggest_int("epochs", 1, 4)

    # ---------------- Train/Val split ----------------
    train_texts, val_texts, y_train, y_val = train_test_split(
        df_class["clean"].tolist(),
        df_class["cluster_id_enc"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df_class["cluster_id_enc"]
    )

    num_labels = df_class["cluster_id_enc"].nunique()

    with mlflow.start_run():
        rebind_file_handler(log_path)

        # Log parameters
        mlflow.log_params({
            "model_name": model_name,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs
        })

        # ---------------- Train + Eval ----------------
        y_true, logits, model = train_eval_single(
            model_name, lr, batch_size, epochs,
            train_texts, val_texts,
            y_train, y_val,
            num_labels
        )

        # ---------------- Compute metrics ----------------
        metrics = evaluate_predictions_full(y_true, logits, num_labels)

        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(k, float(v))

        # Confusion matrix plot
        # preds = logits.argmax(axis=1)
        # cm = confusion_matrix(y_true, preds)

        # plt.figure(figsize=(8, 6))
        # sns.heatmap(cm, annot=False, cmap="Blues")
        # cm_path = f"cm_trial_{trial.number}.png"
        # plt.savefig(cm_path)
        # mlflow.log_artifact(cm_path)
        # plt.close()

        # Save model
        mlflow.pytorch.log_model(model, "model")

    # Save metrics in trial attributes (for summary table)
    trial.set_user_attr("full_metrics", metrics)

    # Optuna optimizes ACCURACY
    return metrics["acc"]


# =============================================================
# MAIN OPTUNA SEARCH RUNNER
# =============================================================
def run_hyperparameter_search(df_class):
    """
    Run Optuna search AND produce full scientific evaluation summary.
    df_class is REQUIRED — no global state.
    """
    mlflow.set_experiment("ArXiv_Classifier_Optimisation")

    study = optuna.create_study(direction="maximize")

    # Wrap objective so df_class is passed in
    def wrapped_objective(trial):
        return objective(trial, df_class)

    study.optimize(wrapped_objective, n_trials=12, show_progress_bar=True)

    logger.info("=== BEST TRIAL PARAMETERS ===")
    logger.info(study.best_trial.params)
    logger.info("Best Accuracy: %.4f", study.best_value)

    # -------------------- Full result summary --------------------
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue

        metrics = t.user_attrs.get("full_metrics", {})
        row = {
            "trial_id": t.number,
            "value": t.value,
            **t.params,
            **metrics
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Log to file
    logger.info("\n=========== ALL TRIAL RESULTS ===========\n%s",
                df.to_string(index=False))

    # Save CSV
    df.to_csv("optuna_trial_results.csv", index=False)
    logger.info("Saved full Optuna results to optuna_trial_results.csv")

    return study
