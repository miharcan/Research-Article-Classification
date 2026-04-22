import copy
import os
import time

import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models.classifier import ClusterDataset
from models.evaluation import evaluate_predictions_full
from models.text_selection import select_texts_for_classification
from utils.config import (
    CLASSIFICATION_CANDIDATES,
    DETERMINISTIC,
    DEVICE,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    HOLDOUT_TEST_SIZE,
    HOLDOUT_VAL_SIZE,
    SEED,
    n_train,
)
from utils.logging_utils import log_path, logger, rebind_file_handler
from utils.reproducibility import set_global_seed
from utils.splits import prepare_holdout_splits


def _loader_workers() -> int:
    override = os.getenv("RAC_DATALOADER_WORKERS")
    if override is not None:
        try:
            return max(0, int(override))
        except ValueError:
            logger.warning("Invalid RAC_DATALOADER_WORKERS=%r; ignoring override", override)
    # Deterministic mode is more prone to DataLoader worker deadlocks in long runs.
    # Prefer single-process loading unless explicitly overridden.
    if DETERMINISTIC:
        return 0
    cpu = os.cpu_count() or 1
    return min(4, max(0, cpu - 1))


def _trial_timeout_seconds() -> int:
    raw = os.getenv("RAC_TRIAL_TIMEOUT_SECONDS", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Invalid RAC_TRIAL_TIMEOUT_SECONDS=%r; disabling timeout", raw)
        return 0


def _check_trial_timeout(
    started_at: float,
    timeout_seconds: int,
    model_name: str,
    phase: str,
    epoch: int,
    epochs: int,
) -> None:
    if timeout_seconds <= 0:
        return
    elapsed = time.monotonic() - started_at
    if elapsed > timeout_seconds:
        raise TimeoutError(
            f"Trial timed out after {elapsed:.1f}s (limit={timeout_seconds}s) "
            f"model={model_name} phase={phase} epoch={epoch}/{epochs}"
        )


def _evaluate_logits_and_loss(
    model: AutoModelForSequenceClassification,
    val_loader: DataLoader,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    trues: list[int] = []
    all_logits: list[np.ndarray] = []
    loss_sum = 0.0
    n_batches = 0

    model.eval()
    with torch.inference_mode():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                loss = F.cross_entropy(logits, labels)

            loss_sum += float(loss.item())
            n_batches += 1
            all_logits.append(logits.cpu().numpy())
            trues.extend(batch["labels"].numpy())

    logits_full = np.vstack(all_logits)
    val_loss = loss_sum / max(1, n_batches)
    return np.array(trues), logits_full, val_loss


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
    early_stopping: bool = True,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = EARLY_STOPPING_MIN_DELTA,
) -> tuple[np.ndarray, np.ndarray, AutoModelForSequenceClassification]:
    """Train one classifier config and return all logits + labels."""
    set_global_seed(seed, deterministic=DETERMINISTIC)
    offline = os.getenv("HF_HUB_OFFLINE", "0") == "1"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        local_files_only=offline,
    )

    train_ds = ClusterDataset(train_texts, y_train, tokenizer)
    val_ds = ClusterDataset(val_texts, y_val, tokenizer)

    g = torch.Generator()
    g.manual_seed(seed)
    workers = _loader_workers()
    timeout_seconds = _trial_timeout_seconds()
    trial_started_at = time.monotonic()
    pin_memory = DEVICE.startswith("cuda")
    logger.info(
        "Train setup: model=%s batch=%d epochs=%d workers=%d timeout_seconds=%d device=%s",
        model_name,
        batch_size,
        epochs,
        workers,
        timeout_seconds,
        DEVICE,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        local_files_only=offline,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    use_amp = DEVICE.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -np.inf
    epochs_no_improve = 0
    epochs_ran = 0

    # ---------------- TRAIN ----------------
    for ep in range(epochs):
        epochs_ran += 1
        model.train()
        for batch in train_loader:
            _check_trial_timeout(
                trial_started_at,
                timeout_seconds,
                model_name,
                phase="train",
                epoch=ep + 1,
                epochs=epochs,
            )
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            scaler.scale(out.loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if early_stopping:
            y_true_ep, logits_ep, val_loss_ep = _evaluate_logits_and_loss(
                model,
                val_loader,
                use_amp=use_amp,
            )
            _check_trial_timeout(
                trial_started_at,
                timeout_seconds,
                model_name,
                phase="eval",
                epoch=ep + 1,
                epochs=epochs,
            )
            metrics_ep = evaluate_predictions_full(y_true_ep, logits_ep, num_labels)
            val_f1_ep = float(metrics_ep["f1_macro"])
            improved = val_f1_ep > (best_val_f1 + early_stopping_min_delta)
            logger.info(
                "[ES] epoch=%d/%d val_f1_macro=%.6f val_loss=%.6f improved=%s",
                ep + 1,
                epochs,
                val_f1_ep,
                val_loss_ep,
                improved,
            )
            if improved:
                best_val_f1 = val_f1_ep
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
                    logger.info(
                        "[ES] stopping at epoch %d (patience=%d, min_delta=%.6f)",
                        ep + 1,
                        early_stopping_patience,
                        early_stopping_min_delta,
                    )
                    break

    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    logger.info("Training completed: epochs_ran=%d requested_epochs=%d", epochs_ran, epochs)

    # ---------------- EVAL ----------------
    y_true, logits_full, _ = _evaluate_logits_and_loss(model, val_loader, use_amp=use_amp)
    return y_true, logits_full, model


# =============================================================
# OPTUNA OBJECTIVE
# =============================================================
def objective(trial: optuna.trial.Trial, split_data: dict, num_labels: int) -> float:
    """Optuna objective on train/val only (test set is untouched)."""

    model_name = trial.suggest_categorical("model_name", CLASSIFICATION_CANDIDATES)
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch", [8, 16, 32])
    epochs = trial.suggest_int("epochs", 1, 8)

    train_texts = split_data["train_texts"]
    val_texts = split_data["val_texts"]
    y_train = split_data["y_train"]
    y_val = split_data["y_val"]

    with mlflow.start_run():
        rebind_file_handler(log_path)

        mlflow.log_params(
            {
                "model_name": model_name,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
            }
        )

        try:
            y_true, logits, _ = train_eval_single(
                model_name,
                lr,
                batch_size,
                epochs,
                train_texts,
                val_texts,
                y_train,
                y_val,
                num_labels,
                seed=SEED,
                early_stopping=True,
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
            )
        except TimeoutError as exc:
            logger.warning("Pruning timed-out trial: %s", exc)
            raise optuna.exceptions.TrialPruned(str(exc)) from exc

        metrics = evaluate_predictions_full(y_true, logits, num_labels)
        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(k, float(v))

    trial.set_user_attr("full_metrics", metrics)

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
        test_size=HOLDOUT_TEST_SIZE,
        val_size=HOLDOUT_VAL_SIZE,
    )
    num_labels = df_class["cluster_id_enc"].nunique()
    study = optuna.create_study(direction="maximize")

    if n_train == 1 and len(CLASSIFICATION_CANDIDATES) == 1:
        # Smoke-style fast path: avoid random slow hyperparameter picks.
        study.enqueue_trial(
            {
                "model_name": CLASSIFICATION_CANDIDATES[0],
                "lr": 5e-6,
                "batch": 32,
                "epochs": 1,
            }
        )
        logger.info("Enqueued fast single-trial params for smoke execution.")

    def wrapped_objective(trial):
        return objective(trial, split_data, num_labels)

    study.optimize(wrapped_objective, n_trials=n_train, show_progress_bar=True)

    logger.info("=== BEST TRIAL PARAMETERS ===")
    logger.info(study.best_trial.params)
    logger.info("Best Macro-F1: %.4f", study.best_value)

    # Final untouched holdout evaluation (train+val -> test): no early stopping on test.
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
        early_stopping=False,
    )
    test_metrics = evaluate_predictions_full(y_true_test, logits_test, num_labels)
    study.set_user_attr("holdout_test_metrics", test_metrics)
    logger.info("=== HOLDOUT TEST METRICS (UNTOUCHED) ===")
    logger.info(test_metrics)

    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue

        metrics = t.user_attrs.get("full_metrics", {})
        row = {
            "trial_id": t.number,
            "value": t.value,
            **t.params,
            **metrics,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info("\n=========== ALL TRIAL RESULTS ===========\n%s", df.to_string(index=False))

    return study
