from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from run_benchmark_multiseed import _export_tables


RUN_FILE_RE = re.compile(r"(?P<combo_id>.+)__seed_(?P<seed>\d+)\.json$")
COMBO_RE = re.compile(r"^cluster_(?P<cluster>.+)__class_(?P<class>.+)__sel_(?P<sel>.+)$")


def _to_row(
    *,
    payload: dict[str, Any],
    combo_id: str,
    seed: int,
    cluster_mode: str,
    class_mode: str,
    selection_mode: str,
) -> dict[str, Any]:
    best_trial = payload.get("best_trial", {}) or {}
    best_params = best_trial.get("params", {}) or {}
    test_metrics = payload.get("holdout_test_metrics", {}) or {}
    best_pipeline = payload.get("best_pipeline", {}) or {}

    row: dict[str, Any] = {
        "combo_id": combo_id,
        "text_representation_cluster": cluster_mode,
        "text_representation_class": class_mode,
        "cluster_selection_mode": selection_mode,
        "seed": seed,
        "started_at": "",
        "ended_at": "",
        "status": "completed",
        "elapsed_seconds": None,
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


def rebuild_seed_dir(seed_dir: Path) -> tuple[int, int]:
    runs_dir = seed_dir / "runs"
    run_files = sorted(runs_dir.glob("*.json"))

    per_seed_rows: list[dict[str, Any]] = []
    clustering_long_rows: list[dict[str, Any]] = []
    skipped = 0

    for p in run_files:
        m = RUN_FILE_RE.match(p.name)
        if not m:
            skipped += 1
            continue
        combo_id = m.group("combo_id")
        seed = int(m.group("seed"))
        mc = COMBO_RE.match(combo_id)
        if not mc:
            skipped += 1
            continue
        cluster_mode = mc.group("cluster")
        class_mode = mc.group("class")
        selection_mode = mc.group("sel")

        payload = json.loads(p.read_text(encoding="utf-8"))
        per_seed_rows.append(
            _to_row(
                payload=payload,
                combo_id=combo_id,
                seed=seed,
                cluster_mode=cluster_mode,
                class_mode=class_mode,
                selection_mode=selection_mode,
            )
        )

        for r in payload.get("clustering_comparison", []) or []:
            clustering_long_rows.append(
                {
                    "combo_id": combo_id,
                    "text_representation_cluster": cluster_mode,
                    "text_representation_class": class_mode,
                    "cluster_selection_mode": selection_mode,
                    "seed": seed,
                    **r,
                }
            )

    if per_seed_rows:
        _export_tables(seed_dir, per_seed_rows, clustering_long_rows)
    return len(per_seed_rows), skipped


def build_paper_tables(results_root: Path, seed_dirs: list[Path]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_root / f"paper_tables_{stamp}_seeds40_44_rebuild"
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for d in seed_dirs:
        p = d / "per_seed_summary.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["source_run"] = d.name
            dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(out_dir / "per_seed_summary_all.csv", index=False)

    id_cols = [
        "combo_id",
        "text_representation_cluster",
        "text_representation_class",
        "cluster_selection_mode",
    ]
    num_cols = [
        c
        for c in all_df.columns
        if c.startswith("test_")
        or c in {"best_trial_value_f1_macro_val", "best_pipeline_score", "elapsed_seconds"}
    ]
    agg = all_df.groupby(id_cols, dropna=False)[num_cols].agg(["mean", "std", "min", "max", "count"]).reset_index()
    agg.columns = ["_".join([str(x) for x in col if x != ""]).strip("_") for col in agg.columns.to_flat_index()]
    agg.to_csv(out_dir / "table_classification_holdout_aggregate_multiseed.csv", index=False)

    top10 = agg.sort_values("test_acc_mean", ascending=False).head(10)
    top10.to_csv(out_dir / "table_classification_top10_multiseed.csv", index=False)

    best_by_rep = (
        agg.sort_values(["text_representation_class", "test_acc_mean"], ascending=[True, False])
        .groupby("text_representation_class", dropna=False)
        .head(1)
        .reset_index(drop=True)
    )
    summary = pd.DataFrame(
        {
            "class_representation": best_by_rep["text_representation_class"],
            "best_combo_id": best_by_rep["combo_id"],
            "cluster_representation": best_by_rep["text_representation_cluster"],
            "accuracy_mean": best_by_rep["test_acc_mean"],
            "accuracy_std": best_by_rep["test_acc_std"],
            "f1_macro_mean": best_by_rep["test_f1_macro_mean"],
            "f1_macro_std": best_by_rep["test_f1_macro_std"],
            "top3_acc_mean": best_by_rep["test_top3_acc_mean"],
            "roc_auc_macro_ovr_mean": best_by_rep["test_roc_auc_macro_ovr_mean"],
            "num_seed_values": best_by_rep["test_acc_count"],
        }
    )
    summary.to_csv(out_dir / "table_paper_results_summary_multiseed.csv", index=False)

    coverage = (
        all_df.groupby(id_cols, dropna=False)["seed"]
        .nunique()
        .reset_index(name="num_seeds_present")
        .sort_values(["num_seeds_present", "combo_id"])
    )
    seed_lists = (
        all_df.groupby(id_cols, dropna=False)["seed"]
        .apply(lambda x: ",".join(str(int(v)) for v in sorted(set(x))))
        .reset_index(name="seeds_present")
    )
    coverage = coverage.merge(seed_lists, on=id_cols, how="left")
    coverage.to_csv(out_dir / "table_run_coverage_by_combo.csv", index=False)

    cdfs = []
    for d in seed_dirs:
        p = d / "clustering_comparison_long.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["source_run"] = d.name
            cdfs.append(df)
    cl_all = pd.concat(cdfs, ignore_index=True)
    cl_all.to_csv(out_dir / "clustering_comparison_long_all.csv", index=False)

    cl_id = id_cols + ["embedding", "algorithm"]
    cl_metrics = [c for c in ["score", "ari", "nmi", "silhouette", "noise_frac", "clusters"] if c in cl_all.columns]
    cl_agg = cl_all.groupby(cl_id, dropna=False)[cl_metrics].agg(["mean", "std", "min", "max", "count"]).reset_index()
    cl_agg.columns = ["_".join([str(x) for x in col if x != ""]).strip("_") for col in cl_agg.columns.to_flat_index()]
    cl_agg.to_csv(out_dir / "table_clustering_by_embedding_algorithm_multiseed.csv", index=False)

    # Journal-style clustering table (new metrics only).
    repr_order = ["abstract", "triples", "abstract_triples", "hybrid"]
    repr_label = {
        "abstract": "Full Abstract",
        "triples": "Triples Only",
        "abstract_triples": "Abstract+Triples",
        "hybrid": "Hybrid Approach",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{l l l c c c c c c}",
        r"\toprule",
        r"Representation & Best Model (Algorithm) & Class Input & K/MinClust & ARI ($\mu\pm\sigma$) & NMI ($\mu\pm\sigma$) & Silh. ($\mu\pm\sigma$) & Noise Frac. & N \\",
        r"\midrule",
    ]
    for rep in repr_order:
        cand = cl_agg[cl_agg["text_representation_cluster"] == rep].copy()
        cand = cand.sort_values(["silhouette_mean", "ari_mean", "nmi_mean"], ascending=False)
        if cand.empty:
            continue
        b = cand.iloc[0]
        k_or_m = "--"
        if pd.notna(b.get("clusters_mean")):
            k_or_m = f"{float(b['clusters_mean']):.1f}"
        noise = "--" if pd.isna(b.get("noise_frac_mean")) else f"{float(b['noise_frac_mean']):.4f}"
        ari = "--" if pd.isna(b.get("ari_mean")) else f"{float(b['ari_mean']):.4f}\\pm{float(b.get('ari_std', 0.0) or 0.0):.4f}"
        nmi = "--" if pd.isna(b.get("nmi_mean")) else f"{float(b['nmi_mean']):.4f}\\pm{float(b.get('nmi_std', 0.0) or 0.0):.4f}"
        sil = "--" if pd.isna(b.get("silhouette_mean")) else f"{float(b['silhouette_mean']):.4f}\\pm{float(b.get('silhouette_std', 0.0) or 0.0):.4f}"
        n = int(b.get("silhouette_count", 0)) if pd.notna(b.get("silhouette_count")) else 0
        lines.append(
            f"{repr_label[rep]} & {b['embedding']} ({b['algorithm']}) & {b['text_representation_class']} & {k_or_m} & {ari} & {nmi} & {sil} & {noise} & {n} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Best clustering outcomes from multi-seed evaluation (seeds 40--44), selected per clustering representation by highest mean silhouette; ARI/NMI included from rebuilt run exports.}",
        r"\label{tab:clus_results_multiseed_best}",
        r"\end{table*}",
    ]
    (out_dir / "table_clustering_best_multiseed_only.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_seed_dirs": [d.name for d in seed_dirs],
        "num_rows_per_seed_summary_all": int(len(all_df)),
        "num_clustering_rows_all": int(len(cl_all)),
    }
    (out_dir / "manifest_paper_tables.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild per-seed and paper tables from run JSON files.")
    p.add_argument("--results-root", default="results", help="Path to results root directory.")
    p.add_argument("--seed-glob", default="*seed4[0-4]", help="Glob under results-root for seed folders.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    seed_dirs = sorted([p for p in results_root.glob(args.seed_glob) if p.is_dir()])
    if not seed_dirs:
        raise FileNotFoundError(f"No seed dirs matched {args.seed_glob} under {results_root}")

    rebuilt = []
    for d in seed_dirs:
        done, skipped = rebuild_seed_dir(d)
        rebuilt.append({"seed_dir": d.name, "rebuilt_rows": done, "skipped_files": skipped})
        print(f"[seed] {d.name}: rebuilt_rows={done} skipped_files={skipped}")

    paper_dir = build_paper_tables(results_root, seed_dirs)
    print(f"[paper] wrote consolidated tables to: {paper_dir}")
    print(json.dumps({"rebuilt": rebuilt, "paper_dir": str(paper_dir)}, indent=2))


if __name__ == "__main__":
    main()

