"""
Utility functions for the obesity classification project.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


CUSTOM_PALETTE = ["#34495E", "#95A5A6", "#5DADE2", "#2C3E50"]


def format_plot(
    ax,
    title=None,
    xlabel=None,
    ylabel=None,
    rotation=0,
    show_grid=True
):
    """
    Apply standard formatting to matplotlib/seaborn plots.
    """
    if title:
        ax.set_title(title, pad=14)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.tick_params(axis="x", rotation=rotation)

    if show_grid:
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.grid(False)

    sns.despine(ax=ax)
    return ax


def plot_correlation_matrix(
    df,
    columns=None,
    title="Feature Correlation Matrix",
    figsize=(9, 8),
    cmap="coolwarm"
):
    """
    Plot a lower-triangle correlation matrix for numeric variables.
    """
    if columns is None:
        corr_df = df.select_dtypes(include=["number"])
    else:
        corr_df = df[columns]

    corr_matrix = corr_df.corr()

    mask = np.triu(
        np.ones_like(corr_matrix, dtype=bool),
        k=1
    )

    plt.figure(figsize=figsize)

    ax = sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        square=True,
        annot_kws={"size": 10, "weight": "bold"},
        cbar_kws={"shrink": 0.8, "label": "Correlation"}
    )

    ax.set_title(title, pad=20, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_class_distribution(
    df,
    target_col,
    order=None,
    title="Class Distribution",
    xlabel="Class",
    ylabel="Number of Observations",
    figsize=(12, 6)
):
    """
    Plot the frequency distribution of a categorical target variable.
    """
    palette = sns.color_palette("Blues", n_colors=len(order)) if order else CUSTOM_PALETTE

    plt.figure(figsize=figsize)

    ax = sns.countplot(
        data=df,
        x=target_col,
        hue=target_col,
        order=order,
        palette=palette,
        legend=False
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%d",
            label_type="edge",
            padding=3,
            fontsize=10
        )

    format_plot(
        ax=ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        rotation=30,
        show_grid=True
    )

    plt.xticks(rotation=30, ha="right")
    plt.show()


def plot_percentage_distribution(
    df,
    target_col,
    order=None,
    title="Percentage Distribution",
    xlabel="Class",
    ylabel="Percentage (%)",
    figsize=(12, 6)
):
    """
    Plot the percentage distribution of a categorical variable.
    """
    percentage_df = (
        df[target_col]
        .value_counts(normalize=True)
        .mul(100)
        .reindex(order)
        .fillna(0)
        .round(2)
        .rename("Percentage")
        .reset_index()
        .rename(columns={"index": target_col})
    )

    palette = sns.color_palette("Blues", n_colors=len(order)) if order else CUSTOM_PALETTE

    plt.figure(figsize=figsize)

    ax = sns.barplot(
        data=percentage_df,
        x=target_col,
        y="Percentage",
        hue=target_col,
        order=order,
        palette=palette,
        legend=False
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f%%",
            label_type="edge",
            padding=3,
            fontsize=10
        )

    format_plot(
        ax=ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        rotation=30,
        show_grid=True
    )

    plt.xticks(rotation=30, ha="right")
    plt.show()

    return percentage_df