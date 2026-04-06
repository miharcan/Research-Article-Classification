import torch
import optuna
import mlflow
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch import optim

from models.classifier import ClusterDataset
from models.evaluation import evaluate_predictions_full
from utils.logging_utils import logger, rebind_file_handler, log_path
from utils.config import DEVICE, CLASSIFICATION_CANDIDATES, n_train, SEED, DETERMINISTIC
from models.text_selection import select_texts_for_classification
from utils.reproducibility import set_global_seed
from utils.splits import prepare_holdout_splits


# =============================================================
# TRAIN + EVAL OF ONE MODEL (used by Optuna)
# =============================================================
def train_eval_single(
    model_name: str,
    lr: float,
    batch_size: int,
    epochs: int,
    train_texts: list[str],
    val_texts: list[str],
    y_train: list[int],
    y_val: list[int],
    num_labels: int,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, AutoModelForSequenceClassification]:
    """Train one classifier config and return all logits + labels."""
    set_global_seed(seed, deterministic=DETERMINISTIC)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = ClusterDataset(train_texts, y_train, tokenizer)
    val_ds   = ClusterDataset(val_texts,   y_val, tokenizer)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
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
    trues = []
    all_logits = []
    
    with torch.no_grad():
        for batch in val_loader:
            logits = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE)
            ).logits

            all_logits.append(logits.cpu().numpy())
            trues.extend(batch["labels"].numpy())

    logits_full = np.vstack(all_logits)

    return np.array(trues), logits_full, model


# =============================================================
# OPTUNA OBJECTIVE
# =============================================================
def objective(trial: optuna.trial.Trial, split_data: dict, num_labels: int) -> float:
    """Optuna objective on train/val only (test set is untouched)."""

    model_name = trial.suggest_categorical("model_name", CLASSIFICATION_CANDIDATES)
    lr         = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch", [8, 16, 32])
    epochs     = trial.suggest_int("epochs", 1, 8)

    train_texts = split_data["train_texts"]
    val_texts = split_data["val_texts"]
    y_train = split_data["y_train"]
    y_val = split_data["y_val"]

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
            num_labels,
            seed=SEED,
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
        # mlflow.pytorch.log_model(model, "model")

    # Save metrics in trial attributes (for summary table)
    trial.set_user_attr("full_metrics", metrics)

    # Optuna optimizes macro-F1 to match research reporting priorities.
    target = metrics.get("f1_macro")
    if target is None:
        raise RuntimeError("f1_macro is missing from evaluation metrics.")
    return float(target)


# =============================================================
# MAIN OPTUNA SEARCH RUNNER
# =============================================================
def run_hyperparameter_search(df_class):
    """
    Run Optuna search AND produce full scientific evaluation summary.
    df_class is REQUIRED — no global state.
    """
    mlflow.set_experiment("ArXiv_Classifier_Optimisation")
    set_global_seed(SEED, deterministic=DETERMINISTIC)

    split_data = prepare_holdout_splits(
        texts=select_texts_for_classification(df_class),
        labels=df_class["cluster_id_enc"].tolist(),
        seed=SEED,
        test_size=0.2,
        val_size=0.2,
    )
    num_labels = df_class["cluster_id_enc"].nunique()
    study = optuna.create_study(direction="maximize")

    # Wrap objective so df_class is passed in
    def wrapped_objective(trial):
        return objective(trial, split_data, num_labels)

    study.optimize(wrapped_objective, n_trials=n_train, show_progress_bar=True)

    logger.info("=== BEST TRIAL PARAMETERS ===")
    logger.info(study.best_trial.params)
    logger.info("Best Macro-F1: %.4f", study.best_value)

    # Final untouched holdout evaluation (train+val -> test).
    best = study.best_trial.params
    final_train_texts = split_data["train_texts"] + split_data["val_texts"]
    final_y_train = split_data["y_train"] + split_data["y_val"]
    y_true_test, logits_test, _ = train_eval_single(
        model_name=best["model_name"],
        lr=best["lr"],
        batch_size=best["batch"],
        epochs=best["epochs"],
        train_texts=final_train_texts,
        val_texts=split_data["test_texts"],
        y_train=final_y_train,
        y_val=split_data["y_test"],
        num_labels=num_labels,
        seed=SEED,
    )
    test_metrics = evaluate_predictions_full(y_true_test, logits_test, num_labels)
    study.set_user_attr("holdout_test_metrics", test_metrics)
    logger.info("=== HOLDOUT TEST METRICS (UNTOUCHED) ===")
    logger.info(test_metrics)

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

    # # Save CSV
    # df.to_csv("optuna_trial_results.csv", index=False)
    # logger.info("Saved full Optuna results to optuna_trial_results.csv")

    return study
