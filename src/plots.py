"""Saving of evaluation plots: detection confusion matrix, precision-recall
curve with AUPR, and the distribution of the shape-match scores.

All figures are rendered with matplotlib on a light surface, using a
colorblind-safe palette (blue for the first series, orange for the second).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from evaluation import EvaluationCounters

# Palette (validated for color-vision deficiency)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
NEUTRAL_CELL = "#f0efec"
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange
SEQUENTIAL_LIGHT = "#cde2fb"
SEQUENTIAL_DARK = "#0d366b"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK_PRIMARY,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
    }
)


def _style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="both", zorder=0)
    ax.set_axisbelow(True)


def plot_confusion_matrix(counters: EvaluationCounters, out_path: Path) -> None:
    """2x2 detection confusion matrix.

    In object detection the true negatives ("background correctly not
    detected") are not defined, so that cell is marked as N/D.
    """
    counts = np.array([[counters.tp, counters.fn], [counters.fp, np.nan]], dtype=float)
    labels = [
        [f"TP\n{counters.tp}", f"FN\n{counters.fn}"],
        [f"FP\n{counters.fp}", "TN\nn/a"],
    ]

    cmap = LinearSegmentedColormap.from_list("seq_blue", [SEQUENTIAL_LIGHT, SEQUENTIAL_DARK])
    max_count = np.nanmax(counts) if np.nanmax(counts) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for row in range(2):
        for col in range(2):
            value = counts[row, col]
            if np.isnan(value):
                face, text_color = NEUTRAL_CELL, INK_MUTED
            else:
                norm = value / max_count
                face = cmap(norm)
                text_color = "#ffffff" if norm > 0.55 else INK_PRIMARY
            ax.add_patch(
                plt.Rectangle((col + 0.02, row + 0.02), 0.96, 0.96, facecolor=face, edgecolor="none")
            )
            ax.text(
                col + 0.5, row + 0.5, labels[row][col],
                ha="center", va="center", fontsize=15, color=text_color, linespacing=1.6,
            )

    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Detected", "Not detected"], fontsize=11, color=INK_SECONDARY)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Sign\npresent", "Sign\nabsent"], fontsize=11, color=INK_SECONDARY)
    ax.tick_params(length=0)
    ax.set_xlabel("Prediction", fontsize=11, labelpad=10)
    ax.set_ylabel("Ground truth", fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Confusion matrix (detection)", fontsize=13, color=INK_PRIMARY, pad=14)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.17)
    fig.text(
        0.5, 0.035, "TN are undefined in object detection (background is not counted)",
        ha="center", va="center", fontsize=8.5, color=INK_MUTED,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _precision_recall_points(counters: EvaluationCounters) -> tuple[np.ndarray, np.ndarray]:
    """PR points obtained by sweeping the matchShapes score threshold.

    Detections are ranked by score (lower = better shape match = higher
    confidence); at each rank precision and recall are computed cumulatively.
    """
    scored = sorted((row[5], row[6]) for row in counters.csv_rows)  # (score, is_tp)
    recalls, precisions = [0.0], [1.0]
    cum_tp = cum_fp = 0
    total = max(counters.total_signs, 1)
    for _, is_tp in scored:
        cum_tp += is_tp
        cum_fp += 1 - is_tp
        recalls.append(min(cum_tp / total, 1.0))
        precisions.append(cum_tp / (cum_tp + cum_fp))
    return np.array(recalls), np.array(precisions)


def average_precision(counters: EvaluationCounters) -> float:
    """AUPR computed as step-wise average precision."""
    recalls, precisions = _precision_recall_points(counters)
    return float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))


def plot_precision_recall(counters: EvaluationCounters, out_path: Path) -> float:
    """Precision-recall curve; returns the AUPR."""
    recalls, precisions = _precision_recall_points(counters)
    aupr = average_precision(counters)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    _style_axes(ax)
    ax.fill_between(recalls, precisions, step="post", color=SEQUENTIAL_LIGHT, zorder=1)
    ax.step(recalls, precisions, where="post", color=SERIES_1, linewidth=2, zorder=2)
    ax.plot(
        recalls[-1], precisions[-1], "o",
        color=SERIES_1, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
    )
    ax.annotate(
        f"matchShapes threshold\n(P {precisions[-1] * 100:.1f}%, R {recalls[-1] * 100:.1f}%)",
        xy=(recalls[-1], precisions[-1]), xytext=(-8, -28),
        textcoords="offset points", ha="right", fontsize=9, color=INK_SECONDARY,
    )

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curve  —  AUPR = {aupr:.3f}", fontsize=13, pad=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return aupr


def plot_score_distribution(counters: EvaluationCounters, out_path: Path) -> None:
    """Histogram of matchShapes scores, split into true and false positives."""
    tp_scores = [row[5] for row in counters.csv_rows if row[6] == 1]
    fp_scores = [row[5] for row in counters.csv_rows if row[6] == 0]
    if not tp_scores and not fp_scores:
        return

    upper = max(tp_scores + fp_scores)
    bins = np.linspace(0, upper, 21)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    _style_axes(ax)
    ax.hist(
        [tp_scores, fp_scores], bins=bins,
        label=["True positives", "False positives"],
        color=[SERIES_1, SERIES_2], edgecolor=SURFACE, linewidth=1.5, zorder=2,
    )
    ax.set_xlabel("matchShapes score (lower = closer to the triangle shape)")
    ax.set_ylabel("Number of detections")
    ax.set_title("Similarity score distribution", fontsize=13, pad=12)
    legend = ax.legend(frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_all_plots(counters: EvaluationCounters, plots_dir: Path) -> float:
    """Save every plot into `plots_dir`; returns the AUPR."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(counters, plots_dir / "confusion_matrix.png")
    aupr = plot_precision_recall(counters, plots_dir / "precision_recall_curve.png")
    plot_score_distribution(counters, plots_dir / "score_distribution.png")
    return aupr
