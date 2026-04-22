from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill clustering metrics (ARI/NMI/silhouette) into existing run JSON exports."
    )
    parser.add_argument(
        "--runs-dir",
        required=True,
        help="Directory with run JSON files (e.g., results/.../runs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed filter. If provided, only files containing '__seed_<seed>.json' are processed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of run JSON files to process (useful for smoke testing).",
    )
    parser.add_argument(
        "--config-default",
        default="config/benchmark.yaml",
        help="Fallback config path when the JSON does not include one.",
    )
    parser.add_argument(
        "--load-n-clustering",
        type=int,
        default=None,
        help="Optional override for clustering subset size (useful for fast smoke validation).",
    )
    return parser.parse_args()


def _iter_run_jsons(runs_dir: Path, seed: int | None) -> list[Path]:
    files = sorted(runs_dir.glob("*.json"))
    if seed is None:
        return files
    token = f"__seed_{seed}.json"
    return [p for p in files if token in p.name]


def _build_overrides(payload: dict, config_default: str, load_n_clustering_override: int | None) -> tuple[str, dict]:
    cfg = payload.get("config", {}) or {}
    cfg_path = cfg.get("config_path") or config_default
    cluster_mode = cfg.get("text_representation_cluster", "hybrid")
    seed = int(cfg.get("seed", 42))
    load_n_clustering = int(cfg.get("load_n_clustering", 5000))
    if load_n_clustering_override is not None:
        load_n_clustering = int(load_n_clustering_override)
    force_k = cfg.get("force_k", None)
    selection_mode = cfg.get("cluster_selection_mode", "unsupervised")

    overrides = {
        "json_path": cfg.get("json_path"),
        "load_n_clustering": load_n_clustering,
        "load_n_classifier": 1,  # clustering-only run path; keep validator happy
        "n_train": 1,  # unused here
        "force_k": force_k,
        "cluster_selection_mode": selection_mode,
        "seed": seed,
        "text_representation_cluster": cluster_mode,
        # keep class mode aligned to avoid unnecessary triple extraction triggers
        "text_representation_class": cluster_mode,
    }
    return cfg_path, overrides


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir).resolve()
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs-dir does not exist: {runs_dir}")

    files = _iter_run_jsons(runs_dir, args.seed)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        print("No run JSON files matched.")
        return

    # Import lazily after path/config parsing.
    from utils.config import configure_runtime
    from utils.reproducibility import set_global_seed
    from data.load_data import load_json_subset, clean_text, top_cat_from_categories
    from data.preprocess import extract_triples_batch

    updated = 0
    failed = 0
    for p in files:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            cfg_path, overrides = _build_overrides(
                payload,
                args.config_default,
                args.load_n_clustering,
            )
            cfg = configure_runtime(config_path=cfg_path, overrides=overrides, require_dataset=True)
            set_global_seed(cfg.seed, cfg.deterministic)

            # Modules import config globals at import time. Reload after runtime
            # configuration so per-run overrides are honored.
            importlib.reload(importlib.import_module("data.load_data"))
            importlib.reload(importlib.import_module("models.embeddings"))
            text_selection_mod = importlib.reload(importlib.import_module("models.text_selection"))
            clustering_mod = importlib.reload(importlib.import_module("models.clustering"))
            prepare_text_representations = text_selection_mod.prepare_text_representations
            compare_embeddings_and_clusterers = clustering_mod.compare_embeddings_and_clusterers
            select_best_pipeline = clustering_mod.select_best_pipeline

            # Build clustering-only frame directly from cfg values so runtime overrides
            # are honored even when modules import config constants at import time.
            total_needed = cfg.load_n_clustering + max(1, cfg.load_n_classifier)
            df_all = load_json_subset(cfg.json_path, total_needed)
            df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)
            df_all["clean"] = df_all["abstract"].astype(str).apply(clean_text)

            needs_triples = cfg.text_representation_cluster in {"triples", "abstract_triples", "hybrid"}
            if needs_triples:
                df_all["triples"] = extract_triples_batch(df_all["abstract"].astype(str).tolist())
            else:
                df_all["triples"] = ""
            df_all["top_category"] = df_all["categories"].astype(str).apply(top_cat_from_categories)

            df_cluster = df_all.iloc[: cfg.load_n_clustering].reset_index(drop=True)
            df_cluster = prepare_text_representations(df_cluster, mode=cfg.text_representation_cluster)

            compare_df, best_k_km, best_k_gmm = compare_embeddings_and_clusterers(df_cluster)
            best_pipeline = select_best_pipeline(compare_df, n_samples=len(df_cluster))

            payload["best_k_kmeans"] = best_k_km
            payload["best_k_gmm"] = best_k_gmm
            payload["best_pipeline"] = best_pipeline
            payload["clustering_comparison"] = compare_df.to_dict(orient="records")

            p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            updated += 1
            print(f"[OK] {p.name}")
        except Exception as e:  # keep backfill resilient per-file
            failed += 1
            print(f"[FAIL] {p.name}: {e}")

    print(
        json.dumps(
            {
                "runs_dir": str(runs_dir),
                "processed": len(files),
                "updated": updated,
                "failed": failed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
