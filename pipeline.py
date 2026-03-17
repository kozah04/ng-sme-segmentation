from __future__ import annotations

import argparse
from pathlib import Path

from src.cluster import DEFAULT_SEGMENTED_DATA_PATH, DEFAULT_FIGURES_DIR, run_clustering_pipeline
from src.features import DEFAULT_CLEAN_DATA_PATH, build_clean_dataset
from src.loader import DEFAULT_RAW_DATA_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Nigerian SME segmentation pipeline.")
    parser.add_argument("--raw-data", type=Path, default=DEFAULT_RAW_DATA_PATH, help="Path to raw .dta data file.")
    parser.add_argument(
        "--clean-output",
        type=Path,
        default=DEFAULT_CLEAN_DATA_PATH,
        help="Path to save cleaned feature dataset (CSV).",
    )
    parser.add_argument(
        "--segmented-output",
        type=Path,
        default=DEFAULT_SEGMENTED_DATA_PATH,
        help="Path to save segmented dataset (CSV).",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory where figures are saved.",
    )
    parser.add_argument("--clusters", type=int, default=4, help="Number of clusters for K-Means/Hierarchical.")
    parser.add_argument("--k-min", type=int, default=2, help="Minimum k for K-Means metric sweep.")
    parser.add_argument("--k-max", type=int, default=9, help="Maximum k for K-Means metric sweep.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip all plot generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k_min >= args.k_max:
        raise ValueError("--k-min must be strictly smaller than --k-max.")

    clean_df = build_clean_dataset(
        raw_data_path=args.raw_data,
        clean_data_path=args.clean_output,
        figures_dir=args.figures_dir,
        save=True,
        make_plots=not args.skip_plots,
    )

    segmented_df, profile, results = run_clustering_pipeline(
        clean_df=clean_df,
        segmented_data_path=args.segmented_output,
        figures_dir=args.figures_dir,
        n_clusters=args.clusters,
        k_values=range(args.k_min, args.k_max + 1),
        random_state=args.random_state,
        save=True,
        make_plots=not args.skip_plots,
    )

    print(f"Cleaned data saved: {args.clean_output}")
    print(f"Segmented data saved: {args.segmented_output}")
    print(f"Figures directory: {args.figures_dir}")
    print(f"Rows: {len(segmented_df):,} | Features (clean): {clean_df.shape[1]} | Clusters: {args.clusters}")
    print("Cluster sizes:")
    print(segmented_df["cluster"].value_counts().sort_index())
    print("\nInferred segment mapping (cluster -> segment):")
    for cluster_id, segment_name in sorted(results["segment_name_map"].items()):
        print(f"  {cluster_id} -> {segment_name}")
    print("\nSegment sizes:")
    print(segmented_df["segment"].value_counts())
    print("\nK-Means vs Hierarchical metrics:")
    print(results["model_comparison"].to_string(index=False))
    print(f"\nAdjusted Rand Index: {results['adjusted_rand_index']:.4f}")
    print("\nCluster profile (means):")
    print(profile.to_string())


if __name__ == "__main__":
    main()
