import logging, os
import platform
import subprocess
import sys
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", f"run_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"),
              logging.StreamHandler()],
)
logger = logging.getLogger("rac")

def rebind_file_handler(path):
    """Reattach file handler after MLflow resets logging."""
    root = logging.getLogger()

    # Remove all file handlers
    for h in root.handlers[:]:
        if isinstance(h, logging.FileHandler):
            root.removeHandler(h)

    # Add a single one back
    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(fh)


def _safe_version(module_name):
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "unavailable"


def _git_commit_short():
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def log_protocol_metadata(cfg) -> None:
    """Write publication-relevant run metadata to logs."""
    logger.info("=== RUN PROTOCOL ===")
    logger.info(
        "split_protocol=stratified holdout_test_size=%.3f holdout_val_size=%.3f",
        cfg.holdout_test_size,
        cfg.holdout_val_size,
    )
    logger.info(
        "objective=f1_macro n_trials=%d cluster_selection_mode=%s",
        cfg.n_train,
        cfg.cluster_selection_mode,
    )
    logger.info(
        "early_stopping_patience=%d early_stopping_min_delta=%.6f",
        cfg.early_stopping_patience,
        cfg.early_stopping_min_delta,
    )
    logger.info(
        "seed=%d deterministic=%s device=%s",
        cfg.seed,
        cfg.deterministic,
        cfg.device,
    )
    logger.info("spacy_model_requested=%s", cfg.spacy_model)

    logger.info("=== ENVIRONMENT ===")
    logger.info("git_commit=%s", _git_commit_short())
    logger.info("python=%s", sys.version.replace("\n", " "))
    logger.info("platform=%s", platform.platform())
    logger.info(
        "versions torch=%s transformers=%s sentence_transformers=%s sklearn=%s spacy=%s optuna=%s mlflow=%s",
        _safe_version("torch"),
        _safe_version("transformers"),
        _safe_version("sentence_transformers"),
        _safe_version("sklearn"),
        _safe_version("spacy"),
        _safe_version("optuna"),
        _safe_version("mlflow"),
    )
