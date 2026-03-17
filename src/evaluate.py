from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)


def evaluate_kmeans_across_k(
    X_scaled: pd.DataFrame,
    k_values: Iterable[int] = range(2, 10),
    random_state: int = 42,
    n_init: int = 10,
) -> pd.DataFrame:
    """Compute silhouette, CH, and DB metrics for each candidate k."""
    rows: list[dict[str, float | int]] = []
    for k in k_values:
        model = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=random_state)
        labels = model.fit_predict(X_scaled)
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(X_scaled, labels),
                "calinski_harabasz": calinski_harabasz_score(X_scaled, labels),
                "davies_bouldin": davies_bouldin_score(X_scaled, labels),
                "inertia": model.inertia_,
            }
        )

    return pd.DataFrame(rows)


def summarize_silhouette_by_cluster(X_scaled: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Return average silhouette score per cluster and cluster size."""
    sil_values = silhouette_samples(X_scaled, labels)
    summary = (
        pd.DataFrame({"cluster": labels, "silhouette": sil_values})
        .groupby("cluster", as_index=False)
        .agg(silhouette_mean=("silhouette", "mean"), n_samples=("silhouette", "size"))
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    return summary


def compare_cluster_methods(
    X_scaled: pd.DataFrame,
    kmeans_labels: pd.Series,
    hierarchical_labels: pd.Series,
) -> pd.DataFrame:
    """Compare K-Means and hierarchical clustering with the notebook metrics."""
    return pd.DataFrame(
        [
            {
                "metric": "silhouette",
                "kmeans": silhouette_score(X_scaled, kmeans_labels),
                "hierarchical": silhouette_score(X_scaled, hierarchical_labels),
            },
            {
                "metric": "calinski_harabasz",
                "kmeans": calinski_harabasz_score(X_scaled, kmeans_labels),
                "hierarchical": calinski_harabasz_score(X_scaled, hierarchical_labels),
            },
            {
                "metric": "davies_bouldin",
                "kmeans": davies_bouldin_score(X_scaled, kmeans_labels),
                "hierarchical": davies_bouldin_score(X_scaled, hierarchical_labels),
            },
        ]
    )


def agreement_score(kmeans_labels: pd.Series, hierarchical_labels: pd.Series) -> float:
    """Adjusted Rand Index between K-Means and hierarchical labels."""
    return adjusted_rand_score(kmeans_labels, hierarchical_labels)
