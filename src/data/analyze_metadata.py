# -------------------------------------------------
# exploratory data analysis and characteristics of the dataset.
# -------------------------------------------------
# config: set to None to scan entire file (may be large); set integer to limit lines for quicker runs
ANALYZE_LIMIT = None
def analyze_arxiv_metadata(json_path, out_dir, limit=None):
    logger.info("Analyzing dataset: %s (limit=%s)", json_path, str(limit))
    total_lines = 0
    parsed_rows = 0
    have_abstract = 0
    have_categories = 0
    missing_fields = Counter()

    subcat_counter = Counter()
    topcat_counter = Counter()
    abstract_lengths = []

    lengths_csv_path = os.path.join(out_dir, "abstract_lengths.csv")
    # write header for streaming lengths
    with open(lengths_csv_path, "w", encoding="utf-8", newline="") as len_f:
        writer = csv.writer(len_f)
        writer.writerow(["row_index", "abstract_length_chars", "abstract_length_tokens"])
        with open(json_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                total_lines += 1
                try:
                    obj = json.loads(line)
                    parsed_rows += 1
                except Exception as e:
                    logger.debug("JSON parse error at line %d: %s", i, str(e))
                    missing_fields["json_parse_error"] += 1
                    continue

                abstract = obj.get("abstract")
                if abstract and isinstance(abstract, str) and abstract.strip():
                    have_abstract += 1
                    chars = len(abstract)
                    tokens = len(abstract.split())
                    abstract_lengths.append(chars)
                    writer.writerow([parsed_rows - 1, chars, tokens])
                else:
                    missing_fields["abstract_missing_or_empty"] += 1

                cats = obj.get("categories")
                if cats and isinstance(cats, str) and cats.strip():
                    have_categories += 1
                    # categories are space-separated subcategories like "cs.AI cs.LG"
                    parts = cats.strip().split()
                    for p in parts:
                        subcat_counter[p] += 1
                        top = p.split(".")[0] if "." in p else p
                        topcat_counter[top] += 1
                else:
                    missing_fields["categories_missing_or_empty"] += 1

    # summary
    summary = {
        "total_lines_seen": total_lines,
        "json_rows_parsed": parsed_rows,
        "have_abstract": have_abstract,
        "have_categories": have_categories,
        "missing_fields": dict(missing_fields),
        "unique_subcategories": len(subcat_counter),
        "unique_top_categories": len(topcat_counter),
    }

    # save counters to CSV
    subcat_df = pd.DataFrame(subcat_counter.items(), columns=["subcategory", "count"]).sort_values("count", ascending=False)
    topcat_df = pd.DataFrame(topcat_counter.items(), columns=["top_category", "count"]).sort_values("count", ascending=False)
    subcat_csv = os.path.join(out_dir, "subcategories_counts.csv")
    topcat_csv = os.path.join(out_dir, "top_categories_counts.csv")
    subcat_df.to_csv(subcat_csv, index=False)
    topcat_df.to_csv(topcat_csv, index=False)

    # save summary JSON
    summary_path = os.path.join(out_dir, "dataset_summary.json")
    try:
        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump(summary, sf, indent=2)
    except Exception:
        logger.exception("Failed to write summary JSON")

    # Create plots
    try:
        # ensure output directory exists and import plotting libs locally to avoid NameError
        os.makedirs(out_dir, exist_ok=True)
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.set_theme(style="whitegrid")
        # Top-N subcategories barplot
        N_SUB = 40
        top_sub = subcat_df.head(N_SUB)
        plt.figure(figsize=(10, max(4, len(top_sub) * 0.25)))
        sns.barplot(x="count", y="subcategory", data=top_sub, palette="viridis")
        plt.title(f"Top {len(top_sub)} Subcategories")
        plt.tight_layout()
        sub_plot_path = os.path.join(out_dir, "top_subcategories.png")
        plt.savefig(sub_plot_path, dpi=150)
        plt.close()
        logger.info("Saved top subcategories plot: %s", sub_plot_path)

        # Top-N top-level categories barplot
        N_TOP = 40
        top_top = topcat_df.head(N_TOP)
        plt.figure(figsize=(8, max(4, len(top_top) * 0.25)))
        sns.barplot(x="count", y="top_category", data=top_top, palette="magma")
        plt.title(f"Top {len(top_top)} Top-Level Categories")
        plt.tight_layout()
        top_plot_path = os.path.join(out_dir, "top_categories.png")
        plt.savefig(top_plot_path, dpi=150)
        plt.close()
        logger.info("Saved top categories plot: %s", top_plot_path)

        # Abstract length distribution (chars)
        if abstract_lengths:
            arr = np.array(abstract_lengths)
            plt.figure(figsize=(8, 4))
            sns.histplot(arr, bins=100, kde=True)
            if arr.max() > 10000:
                plt.xscale("symlog")
            plt.title("Abstract Length Distribution (chars)")
            plt.tight_layout()
            len_plot_path = os.path.join(out_dir, "abstract_length_hist.png")
            plt.savefig(len_plot_path, dpi=150)
            plt.close()
            logger.info("Saved abstract length histogram: %s", len_plot_path)

            # save basic length statistics
            length_stats = {
                "count": int(arr.size),
                "min": int(arr.min()),
                "max": int(arr.max()),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),
                "percentiles": {
                    "5": float(np.percentile(arr, 5)),
                    "25": float(np.percentile(arr, 25)),
                    "50": float(np.percentile(arr, 50)),
                    "75": float(np.percentile(arr, 75)),
                    "95": float(np.percentile(arr, 95)),
                },
            }
            length_stats_path = os.path.join(out_dir, "abstract_length_stats.json")
            with open(length_stats_path, "w", encoding="utf-8") as ls_f:
                json.dump(length_stats, ls_f, indent=2)
            logger.info("Saved abstract length stats: %s", length_stats_path)
        else:
            logger.warning("No abstract lengths collected; skipping length plots/stats")
    except Exception:
        logger.exception("Failed while generating/saving plots")

    logger.info("Dataset analysis complete. Summary: %s", summary)
    return {
        "summary": summary,
        "subcat_csv": subcat_csv,
        "topcat_csv": topcat_csv,
        "lengths_csv": lengths_csv_path,
        "summary_json": summary_path,
    }