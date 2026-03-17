from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer

from .loader import BINARY_COLS, CONTINUOUS_COLS, ORDINAL_COLS, PROJECT_ROOT, load_and_encode_features

LOG_TRANSFORM_COLS = ["d2", "n3", "l1", "b7"]
LOW_VARIANCE_DROP_COLS = ["k6"]

DEFAULT_CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "sme_clean.csv"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

ORDINAL_LABELS = [
    "Finance obstacle",
    "Electricity obstacle",
    "Tax rates obstacle",
    "Informal sector obstacle",
]

BINARY_LABELS = [
    "Bank account",
    "Has loan",
    "Quality cert",
    "New products",
    "R&D spend",
    "Has website",
    "Foreign tech",
    "Competes informal",
    "Women owners",
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def impute_missing_values(df_encoded: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with notebook rules (median for numeric/ordinal, mode for binary)."""
    df_imputed = df_encoded.copy()

    median_cols = CONTINUOUS_COLS + ORDINAL_COLS
    median_imp = SimpleImputer(strategy="median")
    df_imputed[median_cols] = median_imp.fit_transform(df_imputed[median_cols])

    mode_imp = SimpleImputer(strategy="most_frequent")
    df_imputed[BINARY_COLS] = mode_imp.fit_transform(df_imputed[BINARY_COLS])

    return df_imputed


def apply_log_transforms(df: pd.DataFrame, log_cols: list[str] | None = None) -> pd.DataFrame:
    """Apply log1p to skewed continuous features."""
    cols = LOG_TRANSFORM_COLS if log_cols is None else log_cols
    df_transformed = df.copy()
    for col in cols:
        df_transformed[col] = np.log1p(df_transformed[col])
    return df_transformed


def plot_continuous_distributions(df: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, col in enumerate(CONTINUOUS_COLS):
        axes[i].hist(df[col], bins=30, edgecolor="white", color="steelblue")
        axes[i].set_title(col)
        axes[i].set_xlabel("")

    plt.suptitle("Continuous Features - Raw Distributions", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(figures_dir / "dist_continuous.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ordinal_distributions(df: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for i, (col, label) in enumerate(zip(ORDINAL_COLS, ORDINAL_LABELS)):
        axes[i].hist(df[col], bins=[0, 1, 2, 3, 4, 5], edgecolor="white", color="coral", align="left")
        axes[i].set_title(label)
        axes[i].set_xticks([0, 1, 2, 3, 4])
        axes[i].set_xticklabels(["None", "Minor", "Mod", "Major", "Severe"], rotation=45)

    plt.suptitle("Obstacle Ratings - Distributions", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(figures_dir / "dist_ordinal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_binary_distributions(df: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()

    for i, (col, label) in enumerate(zip(BINARY_COLS, BINARY_LABELS)):
        counts = df[col].value_counts().reindex([0, 1], fill_value=0)
        axes[i].bar(["No", "Yes"], counts.values, color=["salmon", "steelblue"], edgecolor="white")
        axes[i].set_title(label)

    axes[-1].set_visible(False)
    plt.suptitle("Binary Features - Yes/No Counts", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(figures_dir / "dist_binary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_log_transformed(df: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, col in enumerate(LOG_TRANSFORM_COLS):
        axes[i].hist(df[col], bins=30, edgecolor="white", color="steelblue")
        axes[i].set_title(f"{col} (log)")

    plt.suptitle("Log-Transformed Features", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(figures_dir / "dist_log_transformed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 13))

    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        annot_kws={"size": 7},
        ax=ax,
    )

    ax.set_title("Feature Correlation Matrix", fontsize=14, pad=15)
    plt.tight_layout()
    fig.savefig(figures_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sales_growth_distribution(df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["sales_growth"], bins=40, edgecolor="white", color="steelblue")
    ax.set_title("Sales Growth Distribution (log scale)")
    ax.set_xlabel("log(current sales) - log(sales 3yrs ago)")
    plt.tight_layout()
    fig.savefig(figures_dir / "dist_sales_growth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_clean_dataset(
    raw_data_path: str | Path | None = None,
    clean_data_path: str | Path | None = None,
    figures_dir: str | Path | None = None,
    save: bool = True,
    make_plots: bool = True,
) -> pd.DataFrame:
    """
    Reproduce the notebook feature engineering workflow and return cleaned features.
    """
    clean_path = DEFAULT_CLEAN_DATA_PATH if clean_data_path is None else Path(clean_data_path)
    fig_dir = DEFAULT_FIGURES_DIR if figures_dir is None else Path(figures_dir)

    df_encoded = load_and_encode_features(raw_data_path)
    df_imputed = impute_missing_values(df_encoded)

    if make_plots:
        _ensure_dir(fig_dir)
        plot_continuous_distributions(df_imputed, fig_dir)
        plot_ordinal_distributions(df_imputed, fig_dir)
        plot_binary_distributions(df_imputed, fig_dir)

    df_log = apply_log_transforms(df_imputed)

    if make_plots:
        plot_log_transformed(df_log, fig_dir)

    df_no_k6 = df_log.drop(columns=LOW_VARIANCE_DROP_COLS)

    if make_plots:
        plot_correlation_heatmap(df_no_k6, fig_dir)

    df_final = df_no_k6.copy()
    df_final["sales_growth"] = df_final["d2"] - df_final["n3"]
    df_final = df_final.drop(columns=["d2", "n3"])

    if make_plots:
        plot_sales_growth_distribution(df_final, fig_dir)

    if save:
        _ensure_dir(clean_path.parent)
        df_final.to_csv(clean_path, index=False)

    return df_final
