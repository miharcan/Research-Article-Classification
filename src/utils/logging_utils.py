import logging, os
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
