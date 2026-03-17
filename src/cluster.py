from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from .evaluate import (
    agreement_score,
    compare_cluster_methods,
    evaluate_kmeans_across_k,
    summarize_silhouette_by_cluster,
)
from .loader import PROJECT_ROOT

DEFAULT_SEGMENTED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "sme_segmented.csv"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

SEGMENT_NAMES = [
    "Credit-Reliant Traders",
    "Formal Export Leaders",
    "Domestic Mid-Tier",
    "Constrained Micro Firms",
]

RADAR_FEATURES = ["l1", "k3a", "k3bc", "k82a", "b8", "h1", "h8", "c22b", "e6", "d3c", "e11", "k30", "c30a"]
RADAR_LABELS = [
    "Employees",
    "Internal capital",
    "Bank borrowing",
    "Has loan",
    "Quality cert",
    "New products",
    "R&D spend",
    "Has website",
    "Foreign tech",
    "Direct exports",
    "Informal competition",
    "Finance obstacle",
    "Electricity obstacle",
]

KEY_FEATURES = ["k3bc", "k82a", "b8", "c22b", "e6", "d3c", "d12b", "e11", "k30", "c30a"]
KEY_LABELS = [
    "Bank borrowing %",
    "Has loan",
    "Quality cert",
    "Has website",
    "Foreign tech",
    "Direct exports %",
    "Foreign inputs %",
    "Informal competition",
    "Finance obstacle",
    "Electricity obstacle",
]

PLOT_COLORS = ["steelblue", "coral", "seagreen", "mediumpurple", "goldenrod", "teal"]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _cluster_name_map(cluster_ids: Iterable[int]) -> dict[int, str]:
    ids = sorted(int(cluster_id) for cluster_id in cluster_ids)
    return {cluster_id: f"Cluster {cluster_id}" for cluster_id in ids}


def _zscore_feature(profile: pd.DataFrame, feature: str) -> pd.Series:
    """Robust z-score across clusters for one feature; returns zeros if unavailable/constant."""
    if feature not in profile.columns:
        return pd.Series(0.0, index=profile.index, dtype=float)
    values = profile[feature].astype(float)
    std = values.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=profile.index, dtype=float)
    return (values - values.mean()) / std


def infer_segment_name_map(profile: pd.DataFrame) -> tuple[dict[int, str], pd.DataFrame]:
    """
    Infer semantic segment names from cluster profiles.

    Cluster labels (0/1/2/3) are arbitrary from K-Means, so this function assigns names
    by feature patterns each run and returns:
    1. cluster_id -> inferred segment_name mapping
    2. score table (cluster x segment) for transparency/debugging
    """
    cluster_ids = sorted(int(cluster_id) for cluster_id in profile.index.tolist())
    if len(cluster_ids) != 4:
        return _cluster_name_map(cluster_ids), pd.DataFrame(index=cluster_ids)

    score_specs = {
        # High dependence on bank credit/loans.
        "Credit-Reliant Traders": {
            "k3bc": 1.25,
            "k82a": 1.00,
            "k3f": 0.20,
            "d3c": -0.20,
        },
        # High export intensity, foreign links, and formal capabilities.
        "Formal Export Leaders": {
            "d3c": 1.00,
            "d12b": 0.80,
            "b8": 0.85,
            "c22b": 0.75,
            "e6": 0.75,
            "h8": 0.30,
        },
        # Mid-scale domestic firms: moderate scale/growth, less export and less credit-reliant.
        "Domestic Mid-Tier": {
            "l1": 0.55,
            "k3a": 0.35,
            "sales_growth": 0.55,
            "h1": 0.20,
            "d3c": -0.55,
            "d12b": -0.45,
            "k3bc": -0.40,
            "k82a": -0.35,
            "k30": -0.30,
            "c30a": -0.30,
        },
        # Smaller firms facing stronger constraints.
        "Constrained Micro Firms": {
            "l1": -1.00,
            "k3a": -0.45,
            "k30": 0.85,
            "c30a": 0.85,
            "e30": 0.45,
            "d3c": -0.45,
            "k82a": -0.25,
        },
    }

    score_table = pd.DataFrame(index=cluster_ids, columns=SEGMENT_NAMES, dtype=float)
    for segment_name in SEGMENT_NAMES:
        weights = score_specs[segment_name]
        score = pd.Series(0.0, index=profile.index, dtype=float)
        for feature_name, weight in weights.items():
            score = score.add(weight * _zscore_feature(profile, feature_name), fill_value=0.0)
        score_table.loc[cluster_ids, segment_name] = score.loc[cluster_ids].values

    # Hungarian assignment enforces one-to-one mapping between clusters and semantic names.
    cost_matrix = -score_table.to_numpy(dtype=float)
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    cluster_to_segment = {
        int(score_table.index[r]): str(score_table.columns[c]) for r, c in zip(row_idx, col_idx)
    }
    return cluster_to_segment, score_table


def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    values = scaler.fit_transform(df)
    X_scaled = pd.DataFrame(values, columns=df.columns, index=df.index)
    return X_scaled, scaler


def plot_pca_explained_variance(X_scaled: pd.DataFrame, figures_dir: Path) -> tuple[PCA, np.ndarray, np.ndarray]:
    pca_full = PCA()
    pca_full.fit(X_scaled)

    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, len(explained) + 1), explained, alpha=0.6, color="steelblue", label="Individual")
    ax.plot(
        range(1, len(cumulative) + 1),
        cumulative,
        color="coral",
        marker="o",
        markersize=4,
        label="Cumulative",
    )
    ax.axhline(0.80, color="gray", linestyle="--", linewidth=1, label="80% threshold")
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1, label="90% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA - Explained Variance")
    ax.legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "pca_explained_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return pca_full, explained, cumulative


def to_pca_2d(X_scaled: pd.DataFrame) -> tuple[PCA, np.ndarray]:
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    return pca_2d, X_pca_2d


def plot_elbow_curve(metrics_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics_df["k"], metrics_df["inertia"], marker="o", color="steelblue")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("Elbow Method")
    ax.set_xticks(metrics_df["k"].tolist())
    plt.tight_layout()
    fig.savefig(figures_dir / "elbow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kmeans_metrics(metrics_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(metrics_df["k"], metrics_df["silhouette"], marker="o", color="steelblue")
    axes[0].set_title("Silhouette Score (higher = better)")
    axes[0].set_xlabel("k")

    axes[1].plot(metrics_df["k"], metrics_df["calinski_harabasz"], marker="o", color="coral")
    axes[1].set_title("Calinski-Harabasz Index (higher = better)")
    axes[1].set_xlabel("k")

    axes[2].plot(metrics_df["k"], metrics_df["davies_bouldin"], marker="o", color="seagreen")
    axes[2].set_title("Davies-Bouldin Index (lower = better)")
    axes[2].set_xlabel("k")

    plt.suptitle("Cluster Evaluation Metrics - K-Means", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(figures_dir / "kmeans_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_silhouette(X_scaled: pd.DataFrame, labels: pd.Series, figures_dir: Path) -> float:
    unique_clusters = sorted(pd.unique(labels))
    sil_values = silhouette_samples(X_scaled, labels)
    avg_score = silhouette_score(X_scaled, labels)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10

    for i, cluster_id in enumerate(unique_clusters):
        cluster_sil_vals = np.sort(sil_values[np.array(labels == cluster_id)])
        size = len(cluster_sil_vals)
        y_upper = y_lower + size
        color = PLOT_COLORS[i % len(PLOT_COLORS)]

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_sil_vals,
            alpha=0.7,
            color=color,
            label=f"Cluster {cluster_id} (n={size})",
        )
        ax.text(-0.05, y_lower + 0.5 * size, str(cluster_id))
        y_lower = y_upper + 10

    ax.axvline(x=avg_score, color="red", linestyle="--", linewidth=1, label=f"Avg score: {avg_score:.3f}")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title(f"Silhouette Plot - K-Means k={len(unique_clusters)}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(figures_dir / f"silhouette_k{len(unique_clusters)}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return avg_score


def plot_cluster_projection(
    X_pca_2d: np.ndarray,
    pca_2d: PCA,
    labels: pd.Series,
    figures_dir: Path,
) -> None:
    unique_clusters = sorted(pd.unique(labels))
    fig, ax = plt.subplots(figsize=(10, 7))

    labels_arr = np.asarray(labels)
    for i, cluster_id in enumerate(unique_clusters):
        mask = labels_arr == cluster_id
        ax.scatter(
            X_pca_2d[mask, 0],
            X_pca_2d[mask, 1],
            c=PLOT_COLORS[i % len(PLOT_COLORS)],
            label=f"Cluster {cluster_id}",
            alpha=0.6,
            s=30,
            edgecolors="none",
        )

    ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(f"K-Means Clusters (k={len(unique_clusters)}) - PCA 2D Projection")
    ax.legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "clusters_pca_2d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dendrogram(X_scaled: pd.DataFrame, figures_dir: Path) -> None:
    linked = linkage(X_scaled, method="ward")

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        linked,
        truncate_mode="lastp",
        p=20,
        leaf_rotation=45,
        leaf_font_size=10,
        show_contracted=True,
        ax=ax,
    )
    ax.set_title("Hierarchical Clustering Dendrogram (Ward linkage, truncated)")
    ax.set_xlabel("Cluster size (in brackets)")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    fig.savefig(figures_dir / "dendrogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_cluster_profile(df: pd.DataFrame) -> pd.DataFrame:
    profile_df = df.drop(columns=["cluster_hc"], errors="ignore")
    return profile_df.groupby("cluster").mean(numeric_only=True).round(3)


def plot_radar_profiles(
    profile: pd.DataFrame,
    cluster_sizes: pd.Series,
    cluster_name_map: dict[int, str],
    figures_dir: Path,
) -> None:
    feature_pairs = [(f, l) for f, l in zip(RADAR_FEATURES, RADAR_LABELS) if f in profile.columns]
    if not feature_pairs:
        return

    features = [f for f, _ in feature_pairs]
    labels = [l for _, l in feature_pairs]

    profile_radar = profile[features].copy()
    span = profile_radar.max() - profile_radar.min()
    profile_radar = (profile_radar - profile_radar.min()) / span.replace(0, 1)

    n_features = len(features)
    angles = [n / float(n_features) * 2 * np.pi for n in range(n_features)]
    angles += angles[:1]

    n_clusters = len(profile_radar)
    n_cols = 2 if n_clusters > 1 else 1
    n_rows = int(np.ceil(n_clusters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), subplot_kw={"polar": True})
    axes = np.atleast_1d(axes).flatten()

    for i, cluster_id in enumerate(profile_radar.index):
        ax = axes[i]
        values = profile_radar.loc[cluster_id].tolist()
        values += values[:1]
        color = PLOT_COLORS[i % len(PLOT_COLORS)]

        ax.plot(angles, values, color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=8)
        ax.set_ylim(0, 1)

        cluster_name = cluster_name_map.get(cluster_id, f"Cluster {cluster_id}")
        n_members = int(cluster_sizes.get(cluster_id, 0))
        ax.set_title(f"Cluster {cluster_id}: {cluster_name}\n(n={n_members})", size=11, fontweight="bold", pad=15)

    for j in range(n_clusters, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Cluster Profiles - Radar Charts", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(figures_dir / "radar_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_key_feature_comparison(profile: pd.DataFrame, cluster_name_map: dict[int, str], figures_dir: Path) -> None:
    feature_pairs = [(f, l) for f, l in zip(KEY_FEATURES, KEY_LABELS) if f in profile.columns]
    if not feature_pairs:
        return

    features = [f for f, _ in feature_pairs]
    labels = [l for _, l in feature_pairs]

    x = np.arange(len(features))
    n_clusters = len(profile.index)
    width = 0.8 / max(n_clusters, 1)

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, cluster_id in enumerate(profile.index):
        vals = [profile.loc[cluster_id, f] for f in features]
        cluster_name = cluster_name_map.get(cluster_id, f"Cluster {cluster_id}")
        ax.bar(
            x + i * width,
            vals,
            width,
            label=cluster_name,
            color=PLOT_COLORS[i % len(PLOT_COLORS)],
            alpha=0.8,
        )

    ax.set_xticks(x + width * (n_clusters - 1) / 2)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("Key Feature Comparison Across Clusters", fontsize=13)
    ax.legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "cluster_bar_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_clustering_pipeline(
    clean_df: pd.DataFrame,
    segmented_data_path: str | Path | None = None,
    figures_dir: str | Path | None = None,
    n_clusters: int = 4,
    k_values: Iterable[int] = range(2, 10),
    random_state: int = 42,
    save: bool = True,
    make_plots: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """
    Reproduce the modelling notebook workflow and return segmented data.
    """
    segmented_path = DEFAULT_SEGMENTED_DATA_PATH if segmented_data_path is None else Path(segmented_data_path)
    fig_dir = DEFAULT_FIGURES_DIR if figures_dir is None else Path(figures_dir)
    if make_plots:
        _ensure_dir(fig_dir)

    X_scaled, scaler = scale_features(clean_df)

    if make_plots:
        pca_full, explained, cumulative = plot_pca_explained_variance(X_scaled, fig_dir)
    else:
        pca_full = PCA().fit(X_scaled)
        explained = pca_full.explained_variance_ratio_
        cumulative = np.cumsum(explained)

    pca_2d, X_pca_2d = to_pca_2d(X_scaled)

    metrics_df = evaluate_kmeans_across_k(X_scaled, k_values=k_values, random_state=random_state, n_init=10)
    if make_plots:
        plot_elbow_curve(metrics_df, fig_dir)
        plot_kmeans_metrics(metrics_df, fig_dir)

    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=random_state)
    clustered_df = clean_df.copy()
    clustered_df["cluster"] = km.fit_predict(X_scaled)

    if make_plots:
        avg_silhouette = plot_silhouette(X_scaled, clustered_df["cluster"], fig_dir)
        plot_cluster_projection(X_pca_2d, pca_2d, clustered_df["cluster"], fig_dir)
        plot_dendrogram(X_scaled, fig_dir)
    else:
        avg_silhouette = silhouette_score(X_scaled, clustered_df["cluster"])

    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    clustered_df["cluster_hc"] = hc.fit_predict(X_scaled)

    metrics_comparison = compare_cluster_methods(X_scaled, clustered_df["cluster"], clustered_df["cluster_hc"])
    ari = agreement_score(clustered_df["cluster"], clustered_df["cluster_hc"])
    sil_by_cluster = summarize_silhouette_by_cluster(X_scaled, clustered_df["cluster"])

    profile = build_cluster_profile(clustered_df)
    cluster_name_map, segment_score_table = infer_segment_name_map(profile)
    cluster_sizes = clustered_df["cluster"].value_counts().sort_index()
    if make_plots:
        plot_radar_profiles(profile, cluster_sizes, cluster_name_map, fig_dir)
        plot_key_feature_comparison(profile, cluster_name_map, fig_dir)

    segmented_df = clustered_df.copy()
    segmented_df["segment"] = segmented_df["cluster"].map(cluster_name_map)

    if save:
        _ensure_dir(segmented_path.parent)
        segmented_df.to_csv(segmented_path, index=False)

    results = {
        "scaler": scaler,
        "pca_full": pca_full,
        "pca_2d": pca_2d,
        "pca_explained_variance": explained,
        "pca_cumulative_variance": cumulative,
        "kmeans_model": km,
        "hierarchical_model": hc,
        "kmeans_metrics_by_k": metrics_df,
        "model_comparison": metrics_comparison,
        "adjusted_rand_index": ari,
        "avg_silhouette_kmeans": avg_silhouette,
        "silhouette_by_cluster": sil_by_cluster,
        "segment_name_map": cluster_name_map,
        "segment_score_table": segment_score_table,
    }
    return segmented_df, profile, results
