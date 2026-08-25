"""Plotting helpers used by the EDA notebook and evaluate.py."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 0 = background, 1 = pass, 2 = fail
WAFER_CMAP = plt.matplotlib.colors.ListedColormap(["white", "#c7d9f0", "#d62728"])


def plot_class_distribution(df, label_col="failureType", save_path=None):
    counts = df[label_col].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_yscale("log")
    ax.set_ylabel("count (log scale)")
    ax.set_title("Failure type distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_sample_grid(df, label_col="failureType", classes=None, n_per_class=4, save_path=None):
    """Grid of sample wafer maps, one row per class."""
    if classes is None:
        classes = sorted(df[label_col].dropna().unique())
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(2 * n_per_class, 2 * len(classes)))
    for i, cls in enumerate(classes):
        samples = df[df[label_col] == cls].sample(min(n_per_class, len(df[df[label_col] == cls])))
        for j, (_, row) in enumerate(samples.iterrows()):
            ax = axes[i, j] if len(classes) > 1 else axes[j]
            ax.imshow(np.asarray(row["waferMap"]), cmap=WAFER_CMAP, vmin=0, vmax=2)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(cls, rotation=0, labelpad=40, fontsize=10)
        # label the row even with axis off
        axes[i, 0].text(-0.4, 0.5, cls, transform=axes[i, 0].transAxes, ha="right", va="center")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_confusion_matrix(cm, class_names, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig
