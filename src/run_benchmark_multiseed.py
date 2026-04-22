from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _parse_seeds(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("No seeds provided.")
    return out


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _sanitize_token(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in s)


def _load_sweep_from_config(config_path: Path) -> tuple[list[str], list[str], list[str]]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")

    cluster_vals = _as_list(payload.get("text_representation_cluster", "hybrid"))
    class_vals = _as_list(payload.get("text_representation_class", "hybrid"))
    selection_vals = _as_list(payload.get("cluster_selection_mode", "unsupervised"))
    return cluster_vals, class_vals, selection_vals


def _run_single_seed_combo(
    repo_root: Path,
    python_exe: str,
    config_path: str,
    seed: int,
    cluster_mode: str,
    class_mode: str,
    selection_mode: str,
    export_json_path: Path,
    offline: bool,
    n_trials_override: int | None,
    heartbeat_seconds: int,
) -> tuple[int, float, str]:
    cmd = [
        python_exe,
        "src/main.py",
        cluster_mode,
        class_mode,
        "--config",
        config_path,
        "--cluster-selection-mode",
        selection_mode,
        "--seed",
        str(seed),
        "--export-json",
        str(export_json_path),
    ]
    if n_trials_override is not None:
        cmd.extend(["--n-trials", str(n_trials_override)])

    env = os.environ.copy()
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    while proc.poll() is None:
        elapsed_now = time.time() - t0
        print(
            f"[HEARTBEAT] combo={_sanitize_token(cluster_mode)}->{_sanitize_token(class_mode)} "
            f"sel={selection_mode} seed={seed} elapsed={elapsed_now:.1f}s",
            flush=True,
        )
        time.sleep(max(5, heartbeat_seconds))
    stdout, stderr = proc.communicate()
    elapsed = time.time() - t0
    output = (stdout or "") + "\n" + (stderr or "")
    return int(proc.returncode), elapsed, output


def _flatten_for_row(
    seed: int,
    combo_id: str,
    cluster_mode: str,
    class_mode: str,
    selection_mode: str,
    payload: dict[str, Any],
    elapsed: float,
    started_at: str,
    ended_at: str,
    status: str,
) -> dict[str, Any]:
    best_trial = payload.get("best_trial", {})
    best_params = best_trial.get("params", {})
    test_metrics = payload.get("holdout_test_metrics", {})
    best_pipeline = payload.get("best_pipeline", {})

    row: dict[str, Any] = {
        "combo_id": combo_id,
        "text_representation_cluster": cluster_mode,
        "text_representation_class": class_mode,
        "cluster_selection_mode": selection_mode,
        "seed": seed,
        "started_at": started_at,
        "ended_at": ended_at,
        "status": status,
        "elapsed_seconds": elapsed,
        "best_trial_number": best_trial.get("number"),
        "best_trial_value_f1_macro_val": best_trial.get("value"),
        "best_model_name": best_params.get("model_name"),
        "best_lr": best_params.get("lr"),
        "best_batch": best_params.get("batch"),
        "best_epochs": best_params.get("epochs"),
        "best_pipeline_embedding": best_pipeline.get("embedding"),
        "best_pipeline_algorithm": best_pipeline.get("algorithm"),
        "best_pipeline_score": best_pipeline.get("score"),
    }
    for k, v in test_metrics.items():
        row[f"test_{k}"] = v
    return row


def _export_tables(
    run_dir: Path,
    per_seed_rows: list[dict[str, Any]],
    clustering_long_rows: list[dict[str, Any]],
) -> None:
    per_seed_df = pd.DataFrame(per_seed_rows).sort_values(["combo_id", "seed"])
    per_seed_df.to_csv(run_dir / "per_seed_summary.csv", index=False)

    clustering_df = pd.DataFrame(clustering_long_rows)
    clustering_df.to_csv(run_dir / "clustering_comparison_long.csv", index=False)

    numeric_cols = [
        c
        for c in per_seed_df.columns
        if c.startswith("test_")
        or c in {"best_trial_value_f1_macro_val", "best_pipeline_score", "elapsed_seconds"}
    ]

    table_cls = (
        per_seed_df.groupby(
            ["combo_id", "text_representation_cluster", "text_representation_class", "cluster_selection_mode"],
            dropna=False,
        )[numeric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    table_cls.columns = [
        "_".join([str(x) for x in col if x != ""]).strip("_")
        for col in table_cls.columns.to_flat_index()
    ]
    table_cls.to_csv(run_dir / "table_classification_holdout_aggregate.csv", index=False)

    if not clustering_df.empty:
        metric_cols = [c for c in ["score", "ari", "nmi", "silhouette"] if c in clustering_df.columns]
        table_cluster = (
            clustering_df.groupby(
                [
                    "combo_id",
                    "text_representation_cluster",
                    "text_representation_class",
                    "cluster_selection_mode",
                    "embedding",
                    "algorithm",
                ],
                dropna=False,
            )[metric_cols]
            .agg(["mean", "std"])
            .reset_index()
        )
        table_cluster.columns = [
            "_".join([str(x) for x in col if x != ""]).strip("_")
            for col in table_cluster.columns.to_flat_index()
        ]
        table_cluster.to_csv(run_dir / "table_clustering_by_embedding_algorithm.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed benchmark and export paper tables.")
    parser.add_argument("--config", default="config/benchmark.yaml", help="Path to benchmark config.")
    parser.add_argument("--seeds", default="40,41,42,43,44", help="Comma-separated seed list.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: results/benchmark_YYYYmmdd_HHMMSS).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for subprocess runs.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 for all runs.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately if one seed/config combo fails.",
    )
    parser.add_argument(
        "--n-trials-override",
        type=int,
        default=None,
        help="Optional override for --n-trials passed to main.py (useful for faster benchmark screening).",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=60,
        help="How often to print in-flight progress heartbeat while one run is executing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) if args.output_dir else (repo_root / "results" / f"benchmark_{stamp}")
    _ensure_dir(run_dir)
    _ensure_dir(run_dir / "runs")

    seeds = _parse_seeds(args.seeds)
    cluster_vals, class_vals, selection_vals = _load_sweep_from_config(config_path)
    combos = list(itertools.product(cluster_vals, class_vals, selection_vals))

    manifest = {
        "config": str(config_path),
        "seeds": seeds,
        "python": args.python,
        "offline": args.offline,
        "sweep": {
            "text_representation_cluster": cluster_vals,
            "text_representation_class": class_vals,
            "cluster_selection_mode": selection_vals,
            "num_combos": len(combos),
        },
        "started_at": datetime.now().isoformat(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    per_seed_rows: list[dict[str, Any]] = []
    clustering_long_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for cluster_mode, class_mode, selection_mode in combos:
        combo_id = (
            f"cluster_{_sanitize_token(cluster_mode)}"
            f"__class_{_sanitize_token(class_mode)}"
            f"__sel_{_sanitize_token(selection_mode)}"
        )
        for seed in seeds:
            seed_json = run_dir / "runs" / f"{combo_id}__seed_{seed}.json"
            started_at_dt = datetime.now()
            started_at = started_at_dt.isoformat()
            print(
                f"[START] combo={combo_id} seed={seed} started_at={started_at}",
                flush=True,
            )
            rc, elapsed, output = _run_single_seed_combo(
                repo_root=repo_root,
                python_exe=args.python,
                config_path=str(config_path),
                seed=seed,
                cluster_mode=cluster_mode,
                class_mode=class_mode,
                selection_mode=selection_mode,
                export_json_path=seed_json,
                offline=args.offline,
                n_trials_override=args.n_trials_override,
                heartbeat_seconds=args.heartbeat_seconds,
            )
            ended_at_dt = datetime.now()
            ended_at = ended_at_dt.isoformat()
            (run_dir / "runs" / f"{combo_id}__seed_{seed}.log").write_text(output, encoding="utf-8")

            if rc != 0 or not seed_json.exists():
                status = "failed"
                print(
                    f"[END] combo={combo_id} seed={seed} status={status} ended_at={ended_at} elapsed={elapsed:.2f}s",
                    flush=True,
                )
                failures.append(
                    {
                        "combo_id": combo_id,
                        "seed": seed,
                        "started_at": started_at,
                        "ended_at": ended_at,
                        "status": status,
                        "returncode": rc,
                        "elapsed_seconds": elapsed,
                    }
                )
                if args.fail_fast:
                    break
                continue

            payload = json.loads(seed_json.read_text(encoding="utf-8"))
            status = "completed"
            print(
                f"[END] combo={combo_id} seed={seed} status={status} ended_at={ended_at} elapsed={elapsed:.2f}s",
                flush=True,
            )
            per_seed_rows.append(
                _flatten_for_row(
                    seed=seed,
                    combo_id=combo_id,
                    cluster_mode=cluster_mode,
                    class_mode=class_mode,
                    selection_mode=selection_mode,
                    payload=payload,
                    elapsed=elapsed,
                    started_at=started_at,
                    ended_at=ended_at,
                    status=status,
                )
            )

            for row in payload.get("clustering_comparison", []):
                clustering_long_rows.append(
                    {
                        "combo_id": combo_id,
                        "text_representation_cluster": cluster_mode,
                        "text_representation_class": class_mode,
                        "cluster_selection_mode": selection_mode,
                        "seed": seed,
                        **row,
                    }
                )
        if args.fail_fast and failures:
            break

    if per_seed_rows:
        _export_tables(run_dir=run_dir, per_seed_rows=per_seed_rows, clustering_long_rows=clustering_long_rows)

    result = {
        "run_dir": str(run_dir),
        "num_requested_seeds": len(seeds),
        "num_requested_combos": len(combos),
        "num_requested_runs": len(seeds) * len(combos),
        "num_completed_runs": len(per_seed_rows),
        "num_failed_runs": len(failures),
        "failures": failures,
        "finished_at": datetime.now().isoformat(),
    }
    (run_dir / "result_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
