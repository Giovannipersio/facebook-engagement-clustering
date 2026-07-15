"""
Helper functions for exploratory data analysis and visualization.

This module centralizes reusable functions for EDA in notebook workflows.
The default visual style follows the palette and formatting used in the
project notebooks.
"""

# =============================================================================
# Imports
# =============================================================================

import math
import textwrap

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =============================================================================
# Constants and global settings
# =============================================================================

DEFAULT_PALETTE = ["#34495E", "#95A5A6", "#5DADE2", "#2C3E50"]


# =============================================================================
# Public module API
# =============================================================================

__all__ = [
    "DEFAULT_PALETTE",
    "set_plot_style",
    "plot_correlation_matrix",
    "plot_categorical_distributions",
    "plot_numeric_distributions",
    "plot_categorical_vs_target",
    "plot_numeric_vs_target",
    "summarize_missing_values",
    "plot_missing_values",
    "plot_boxplots",
    "detect_outliers_iqr",
]


# =============================================================================
# Private helper functions
# =============================================================================

def _validate_columns(df, columns):
    """Validate dataframe columns and return them as a list."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if columns is None:
        return df.columns.tolist()

    if isinstance(columns, str):
        columns = [columns]
    else:
        columns = list(columns)

    if not columns:
        raise ValueError("At least one column must be provided.")

    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in dataframe: {missing_columns}")

    return columns


def _validate_numeric_columns(df, columns=None):
    """Validate numeric dataframe columns and return them as a list."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns.tolist()
    else:
        columns = _validate_columns(df, columns)

    non_numeric_columns = [
        column for column in columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric_columns:
        raise ValueError(
            "The following columns must be numeric: "
            f"{non_numeric_columns}"
        )

    if not columns:
        raise ValueError("No numeric columns available.")

    return columns


def _get_palette(palette, n_categories):
    """Return a palette with the requested number of colors."""
    if n_categories <= 0:
        return []

    if palette is None:
        palette = DEFAULT_PALETTE

    if isinstance(palette, str):
        return sns.color_palette(palette, n_colors=n_categories)

    colors = list(palette)
    if not colors:
        colors = DEFAULT_PALETTE.copy()

    if len(colors) >= n_categories:
        return colors[:n_categories]

    repetitions = math.ceil(n_categories / len(colors))
    return (colors * repetitions)[:n_categories]


def _format_tick_labels(labels, replace_underscores=True, wrap=False, wrap_width=14):
    """Format tick labels with optional underscore replacement and wrapping."""
    formatted_labels = []

    for label in labels:
        text = label.get_text() if hasattr(label, "get_text") else str(label)

        if replace_underscores:
            text = text.replace("_", " ")

        if wrap and wrap_width is not None:
            text = textwrap.fill(text, width=wrap_width)

        formatted_labels.append(text)

    return formatted_labels


def _apply_tick_formatting(
    ax,
    xtick_rotation=0,
    ytick_rotation=0,
    format_xticks=True,
    format_yticks=False,
    wrap_xticks=False,
    wrap_yticks=False,
    wrap_width=14,
):
    """Apply standardized tick formatting to an axis."""
    if format_xticks:
        xticks = ax.get_xticks()
        xlabels = _format_tick_labels(
            ax.get_xticklabels(),
            wrap=wrap_xticks,
            wrap_width=wrap_width,
        )
        if len(xticks) == len(xlabels):
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels)

    if format_yticks:
        yticks = ax.get_yticks()
        ylabels = _format_tick_labels(
            ax.get_yticklabels(),
            wrap=wrap_yticks,
            wrap_width=wrap_width,
        )
        if len(yticks) == len(ylabels):
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels)

    ax.tick_params(axis="x", rotation=xtick_rotation)
    ax.tick_params(axis="y", rotation=ytick_rotation)

    if xtick_rotation:
        plt.setp(ax.get_xticklabels(), ha="right")
    else:
        plt.setp(ax.get_xticklabels(), ha="center")


def _format_axis(
    ax,
    title=None,
    xlabel="",
    ylabel="",
    xtick_rotation=0,
    ytick_rotation=0,
    format_xticks=True,
    format_yticks=False,
    wrap_xticks=False,
    wrap_yticks=False,
    wrap_width=14,
    show_grid=False,
    grid_axis="y",
):
    """Apply the shared project style to a matplotlib axis."""
    if title is not None:
        ax.set_title(title, pad=15, weight="bold")

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    _apply_tick_formatting(
        ax=ax,
        xtick_rotation=xtick_rotation,
        ytick_rotation=ytick_rotation,
        format_xticks=format_xticks,
        format_yticks=format_yticks,
        wrap_xticks=wrap_xticks,
        wrap_yticks=wrap_yticks,
        wrap_width=wrap_width,
    )

    if show_grid:
        ax.grid(axis=grid_axis, alpha=0.3)
    else:
        ax.grid(False)

    sns.despine(ax=ax)
    return ax


def _add_bar_labels(ax, fmt="%d", padding=3, fontsize=10):
    """Add labels to all bar containers in an axis."""
    for container in ax.containers:
        if hasattr(ax, "bar_label"):
            ax.bar_label(
                container,
                fmt=fmt,
                padding=padding,
                fontsize=fontsize,
            )
        else:
            orientation = getattr(container, "orientation", "vertical")
            for patch in container.patches:
                if orientation == "horizontal":
                    value = patch.get_width()
                    xy = (value, patch.get_y() + patch.get_height() / 2)
                    xytext = (padding, 0)
                    ha = "left"
                    va = "center"
                else:
                    value = patch.get_height()
                    xy = (patch.get_x() + patch.get_width() / 2, value)
                    xytext = (0, padding)
                    ha = "center"
                    va = "bottom"

                label = fmt % value if isinstance(fmt, str) else str(value)
                ax.annotate(
                    label,
                    xy,
                    ha=ha,
                    va=va,
                    xytext=xytext,
                    textcoords="offset points",
                    fontsize=fontsize,
                )

    ax.margins(x=0.12, y=0.12)


def _get_column_order(order, column):
    """Return the plotting order for a given column."""
    if order is None:
        return None

    if isinstance(order, dict):
        return order.get(column)

    return order


def _count_palette_levels(series, order=None, dropna=True):
    """Return the number of levels needed for a plot palette."""
    if order is not None:
        return len(order)

    return series.nunique(dropna=dropna)


def _apply_category_mappings(df, category_mappings):
    """Apply category mappings without mutating the original dataframe."""
    plot_df = df.copy()

    if category_mappings is None:
        return plot_df

    for column, mapping in category_mappings.items():
        if column not in plot_df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe.")

        mapped_values = plot_df[column].map(mapping)
        plot_df[column] = mapped_values.where(mapped_values.notna(), plot_df[column])

    return plot_df


def _fill_missing_categories(df, columns, missing_label):
    """Replace missing categorical values for plotting only."""
    plot_df = df.copy()

    for column in columns:
        plot_df[column] = plot_df[column].astype("object")
        plot_df[column] = plot_df[column].where(
            plot_df[column].notna(),
            missing_label,
        )

    return plot_df


def _make_grid(nplots, ncols=2, figsize=None, row_height=5, column_width=7):
    """Create a flattened grid of subplots."""
    ncols = min(max(1, ncols), nplots)
    nrows = math.ceil(nplots / ncols)

    if figsize is None:
        figsize = (column_width * ncols, row_height * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes, dtype=object).reshape(-1)

    return fig, axes


# =============================================================================
# Visual configuration
# =============================================================================

def set_plot_style(
    palette=None,
    context="notebook",
    style="white",
    font="sans-serif",
    font_scale=1.0,
    figure_size=(12, 6),
    figure_dpi=100,
    grid_alpha=0.3,
    retina=True,
):
    """
    Apply the default visual style for seaborn and matplotlib plots.

    Parameters
    ----------
    palette : list, str or None, default=None
        Color palette. If None, DEFAULT_PALETTE is used.
    context : str, default="notebook"
        Seaborn context.
    style : str, default="white"
        Seaborn axis style.
    font : str, default="sans-serif"
        Font family.
    font_scale : float, default=1.0
        Font scale factor.
    figure_size : tuple, default=(12, 6)
        Default matplotlib figure size.
    figure_dpi : int, default=100
        Default figure resolution.
    grid_alpha : float, default=0.3
        Grid transparency.
    retina : bool, default=True
        Use retina output when running in IPython/Jupyter.
    """
    if palette is None:
        palette = DEFAULT_PALETTE

    if retina:
        try:
            from IPython import get_ipython

            ipython = get_ipython()
            if ipython is not None:
                ipython.run_line_magic(
                    "config",
                    'InlineBackend.figure_format = "retina"',
                )
        except Exception:
            pass

    sns.set_theme(
        context=context,
        style=style,
        palette=palette,
        font=font,
        font_scale=font_scale,
        rc={
            "figure.figsize": figure_size,
            "figure.dpi": figure_dpi,
            "figure.autolayout": True,
            "grid.alpha": grid_alpha,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
        },
    )


# =============================================================================
# Univariate analysis
# =============================================================================

def plot_categorical_distributions(
    df,
    columns,
    category_mappings=None,
    order=None,
    ncols=2,
    figsize=None,
    palette=None,
    title_suffix="Distribution",
    bar_labels=True,
    label_padding=3,
    label_fontsize=10,
    xtick_rotation=0,
    ytick_rotation=0,
    include_missing=False,
    missing_label="Missing",
    wrap_xticks=False,
    wrap_width=14,
    show=True,
):
    """
    Plot count distributions for categorical or discrete features.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure.
    axes : numpy.ndarray
        Flattened array of axes.
    """
    columns = _validate_columns(df, columns)
    plot_df = _apply_category_mappings(df, category_mappings)

    if include_missing:
        plot_df = _fill_missing_categories(plot_df, columns, missing_label)

    nplots = len(columns)
    fig, axes = _make_grid(
        nplots=nplots,
        ncols=ncols,
        figsize=figsize,
        row_height=5,
        column_width=7,
    )

    for index, column in enumerate(columns):
        ax = axes[index]
        column_order = _get_column_order(order, column)
        n_categories = _count_palette_levels(
            plot_df[column],
            order=column_order,
            dropna=not include_missing,
        )
        current_palette = _get_palette(palette, n_categories)

        sns.countplot(
            data=plot_df,
            x=column,
            hue=column,
            order=column_order,
            hue_order=column_order,
            palette=current_palette,
            dodge=False,
            ax=ax,
        )

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        _format_axis(
            ax=ax,
            title=f"{column} {title_suffix}",
            xlabel="",
            ylabel="",
            xtick_rotation=xtick_rotation,
            ytick_rotation=ytick_rotation,
            wrap_xticks=wrap_xticks,
            wrap_width=wrap_width,
        )

        if bar_labels:
            _add_bar_labels(
                ax=ax,
                fmt="%d",
                padding=label_padding,
                fontsize=label_fontsize,
            )

    for index in range(nplots, len(axes)):
        axes[index].set_visible(False)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes


def plot_numeric_distributions(
    df,
    columns,
    figsize=None,
    palette=None,
    color=None,
    kde=True,
    bins="auto",
    hist_alpha=0.6,
    box_width=0.4,
    show=True,
):
    """
    Plot histogram/KDE and boxplot pairs for continuous numeric features.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure.
    axes : numpy.ndarray
        Axes matrix with one row per feature and two columns.
    """
    columns = _validate_numeric_columns(df, columns)

    if color is None:
        color = _get_palette(palette, 1)[0]

    if figsize is None:
        figsize = (14, 4 * len(columns))

    fig, axes = plt.subplots(len(columns), 2, figsize=figsize)
    axes = np.array(axes, dtype=object).reshape(len(columns), 2)

    for index, column in enumerate(columns):
        ax_hist = axes[index, 0]
        ax_box = axes[index, 1]

        sns.histplot(
            data=df,
            x=column,
            kde=kde,
            bins=bins,
            ax=ax_hist,
            color=color,
            alpha=hist_alpha,
        )
        _format_axis(
            ax=ax_hist,
            title=f"{column} Distribution",
            xlabel="",
            ylabel="Count",
            format_xticks=False,
        )

        sns.boxplot(
            data=df,
            x=column,
            ax=ax_box,
            color=color,
            width=box_width,
            flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
        )
        _format_axis(
            ax=ax_box,
            title=f"{column} Boxplot",
            xlabel="",
            ylabel="",
            format_xticks=False,
        )

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes


# =============================================================================
# Bivariate analysis
# =============================================================================

def plot_categorical_vs_target(
    df,
    columns,
    target,
    category_mappings=None,
    target_mapping=None,
    order=None,
    target_order=None,
    normalize=True,
    ncols=2,
    figsize=None,
    palette=None,
    title_suffix="by Target",
    bar_labels=True,
    label_padding=3,
    label_fontsize=9,
    xtick_rotation=0,
    include_missing=False,
    missing_label="Missing",
    wrap_xticks=False,
    wrap_width=14,
    show=True,
):
    """
    Plot categorical feature distributions split by a target column.

    When normalize=True, bars show the target percentage within each category.
    Otherwise, bars show absolute counts.
    """
    columns = _validate_columns(df, columns)
    _validate_columns(df, target)

    mappings = {} if category_mappings is None else dict(category_mappings)
    if target_mapping is not None:
        mappings[target] = target_mapping

    plot_df = _apply_category_mappings(df, mappings or None)

    if include_missing:
        plot_df = _fill_missing_categories(
            plot_df,
            columns + [target],
            missing_label,
        )

    if target_order is None:
        target_order = plot_df[target].dropna().unique().tolist()

    nplots = len(columns)
    fig, axes = _make_grid(
        nplots=nplots,
        ncols=ncols,
        figsize=figsize,
        row_height=5,
        column_width=7,
    )
    current_palette = _get_palette(palette, len(target_order))

    for index, column in enumerate(columns):
        ax = axes[index]
        column_order = _get_column_order(order, column)

        if normalize:
            plot_data = (
                plot_df
                .groupby([column, target], dropna=not include_missing)
                .size()
                .rename("count")
                .reset_index()
            )
            totals = plot_data.groupby(column)["count"].transform("sum")
            plot_data["percentage"] = np.where(
                totals.gt(0),
                plot_data["count"] / totals * 100,
                0,
            )

            sns.barplot(
                data=plot_data,
                x=column,
                y="percentage",
                hue=target,
                order=column_order,
                hue_order=target_order,
                palette=current_palette,
                ax=ax,
            )
            ylabel = "Percentage (%)"
            label_fmt = "%.1f%%"
        else:
            sns.countplot(
                data=plot_df,
                x=column,
                hue=target,
                order=column_order,
                hue_order=target_order,
                palette=current_palette,
                ax=ax,
            )
            ylabel = "Count"
            label_fmt = "%d"

        _format_axis(
            ax=ax,
            title=f"{column} {title_suffix}",
            xlabel="",
            ylabel=ylabel,
            xtick_rotation=xtick_rotation,
            wrap_xticks=wrap_xticks,
            wrap_width=wrap_width,
        )

        if ax.get_legend() is not None:
            ax.legend(title=target, frameon=False)

        if bar_labels:
            _add_bar_labels(
                ax=ax,
                fmt=label_fmt,
                padding=label_padding,
                fontsize=label_fontsize,
            )

    for index in range(nplots, len(axes)):
        axes[index].set_visible(False)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes


def plot_numeric_vs_target(
    df,
    columns,
    target,
    target_mapping=None,
    target_order=None,
    plot_type="box",
    ncols=2,
    figsize=None,
    palette=None,
    xtick_rotation=0,
    wrap_xticks=False,
    wrap_width=14,
    show=True,
):
    """
    Plot numeric feature distributions grouped by a categorical target.

    plot_type can be "box", "violin" or "strip".
    """
    columns = _validate_numeric_columns(df, columns)
    _validate_columns(df, target)

    plot_df = df.copy()
    if target_mapping is not None:
        mapped_target = plot_df[target].map(target_mapping)
        plot_df[target] = mapped_target.where(mapped_target.notna(), plot_df[target])

    if target_order is None:
        target_order = plot_df[target].dropna().unique().tolist()

    current_palette = _get_palette(palette, len(target_order))
    nplots = len(columns)
    fig, axes = _make_grid(
        nplots=nplots,
        ncols=ncols,
        figsize=figsize,
        row_height=5,
        column_width=7,
    )

    for index, column in enumerate(columns):
        ax = axes[index]

        if plot_type == "box":
            sns.boxplot(
                data=plot_df,
                x=target,
                y=column,
                hue=target,
                order=target_order,
                hue_order=target_order,
                palette=current_palette,
                dodge=False,
                width=0.5,
                flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
                ax=ax,
            )
        elif plot_type == "violin":
            sns.violinplot(
                data=plot_df,
                x=target,
                y=column,
                hue=target,
                order=target_order,
                hue_order=target_order,
                palette=current_palette,
                dodge=False,
                inner="quartile",
                cut=0,
                ax=ax,
            )
        elif plot_type == "strip":
            sns.stripplot(
                data=plot_df,
                x=target,
                y=column,
                hue=target,
                order=target_order,
                hue_order=target_order,
                palette=current_palette,
                dodge=False,
                alpha=0.7,
                jitter=0.2,
                ax=ax,
            )
        else:
            raise ValueError("plot_type must be one of: 'box', 'violin', 'strip'.")

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        _format_axis(
            ax=ax,
            title=f"{column} by {target}",
            xlabel="",
            ylabel=column,
            xtick_rotation=xtick_rotation,
            wrap_xticks=wrap_xticks,
            wrap_width=wrap_width,
        )

    for index in range(nplots, len(axes)):
        axes[index].set_visible(False)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes


# =============================================================================
# Multivariate analysis
# =============================================================================

def plot_correlation_matrix(
    df,
    columns=None,
    encoded_columns=None,
    method="pearson",
    title="Feature Correlation Matrix",
    figsize=(9, 8),
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    mask_upper=True,
    linewidths=0.5,
    vmin=-1,
    vmax=1,
    square=True,
    annot_kws=None,
    cbar_kws=None,
    xtick_rotation=45,
    ytick_rotation=0,
    show=True,
):
    """
    Plot a standardized correlation heatmap for numeric variables.

    encoded_columns can create temporary numeric columns before correlation:
    {
        "Sex": {"mapping": {"male": 0, "female": 1}, "new_column": "Sex_Code"}
    }
    """
    plot_df = df.copy()

    if encoded_columns is not None:
        for source_column, config in encoded_columns.items():
            if source_column not in plot_df.columns:
                raise ValueError(f"Column '{source_column}' not found in dataframe.")

            mapping = config.get("mapping")
            new_column = config.get("new_column", f"{source_column}_Code")

            if mapping is None:
                raise ValueError(
                    f"A mapping must be provided for '{source_column}'."
                )

            plot_df[new_column] = plot_df[source_column].map(mapping)

    if columns is None:
        corr_columns = plot_df.select_dtypes(include=["number"]).columns.tolist()
    else:
        corr_columns = _validate_columns(plot_df, columns)

    corr_columns = _validate_numeric_columns(plot_df, corr_columns)
    corr_df = plot_df[corr_columns]

    if corr_df.empty:
        raise ValueError("No numeric columns available to compute correlation matrix.")

    corr_matrix = corr_df.corr(method=method)

    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    if annot_kws is None:
        annot_kws = {"size": 10, "weight": "bold"}

    if cbar_kws is None:
        cbar_kws = {"shrink": 0.8}

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        linewidths=linewidths,
        vmin=vmin,
        vmax=vmax,
        square=square,
        annot_kws=annot_kws,
        cbar_kws=cbar_kws,
        ax=ax,
    )

    ax.set_title(title, pad=20, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")

    _apply_tick_formatting(
        ax=ax,
        xtick_rotation=xtick_rotation,
        ytick_rotation=ytick_rotation,
        format_xticks=True,
        format_yticks=True,
    )

    plt.tight_layout()

    if show:
        plt.show()

    return ax, corr_matrix


# =============================================================================
# Missing values and data quality
# =============================================================================

def summarize_missing_values(df, columns=None, include_zero=False, sort=True):
    """
    Return a summary table with missing value counts and percentages.
    """
    columns = _validate_columns(df, columns)
    total_rows = len(df)

    summary = pd.DataFrame(
        {
            "missing_count": df[columns].isna().sum(),
            "missing_percent": (
                df[columns].isna().mean().mul(100)
                if total_rows > 0
                else 0
            ),
            "dtype": df[columns].dtypes.astype(str),
        }
    )

    if not include_zero:
        summary = summary[summary["missing_count"] > 0]

    if sort and not summary.empty:
        summary = summary.sort_values(
            by=["missing_count", "missing_percent"],
            ascending=False,
        )

    return summary


def plot_missing_values(
    df,
    columns=None,
    figsize=None,
    palette=None,
    title="Missing Values by Feature",
    label_padding=3,
    label_fontsize=10,
    show=True,
):
    """
    Plot missing value percentages by feature.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Generated axis.
    summary : pandas.DataFrame
        Missing value summary used in the plot.
    """
    summary = summarize_missing_values(
        df=df,
        columns=columns,
        include_zero=False,
        sort=True,
    )

    if figsize is None:
        figsize = (12, max(4, 0.45 * max(len(summary), 1) + 2))

    fig, ax = plt.subplots(figsize=figsize)

    if summary.empty:
        ax.text(
            0.5,
            0.5,
            "No missing values",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            weight="bold",
        )
        ax.set_axis_off()
    else:
        plot_data = (
            summary
            .reset_index()
            .rename(columns={"index": "feature"})
        )
        color = _get_palette(palette, 1)[0]

        sns.barplot(
            data=plot_data,
            x="missing_percent",
            y="feature",
            color=color,
            ax=ax,
        )

        _format_axis(
            ax=ax,
            title=title,
            xlabel="Missing (%)",
            ylabel="",
            format_xticks=False,
            format_yticks=True,
            wrap_yticks=True,
            show_grid=True,
            grid_axis="x",
        )

        _add_bar_labels(
            ax=ax,
            fmt="%.1f%%",
            padding=label_padding,
            fontsize=label_fontsize,
        )

    plt.tight_layout()

    if show:
        plt.show()

    return ax, summary


# =============================================================================
# Outliers
# =============================================================================

def plot_boxplots(
    df,
    columns=None,
    ncols=2,
    figsize=None,
    palette=None,
    color=None,
    title_suffix="Boxplot",
    show=True,
):
    """
    Plot standardized horizontal boxplots for numeric columns.
    """
    columns = _validate_numeric_columns(df, columns)

    if color is None:
        color = _get_palette(palette, 1)[0]

    nplots = len(columns)
    fig, axes = _make_grid(
        nplots=nplots,
        ncols=ncols,
        figsize=figsize,
        row_height=4,
        column_width=7,
    )

    for index, column in enumerate(columns):
        ax = axes[index]

        sns.boxplot(
            data=df,
            x=column,
            color=color,
            width=0.4,
            flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
            ax=ax,
        )

        _format_axis(
            ax=ax,
            title=f"{column} {title_suffix}",
            xlabel="",
            ylabel="",
            format_xticks=False,
        )

    for index in range(nplots, len(axes)):
        axes[index].set_visible(False)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes


def detect_outliers_iqr(df, columns=None, factor=1.5, return_mask=False):
    """
    Detect outliers using the IQR rule.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    columns : list[str], str or None, default=None
        Numeric columns to analyze. If None, all numeric columns are used.
    factor : float, default=1.5
        IQR multiplier used to define lower and upper bounds.
    return_mask : bool, default=False
        If True, also return a boolean dataframe marking outlier rows.

    Returns
    -------
    summary : pandas.DataFrame
        Outlier bounds, counts and percentages by feature.
    outlier_mask : pandas.DataFrame, optional
        Returned only when return_mask=True.
    """
    columns = _validate_numeric_columns(df, columns)
    total_rows = len(df)

    rows = []
    outlier_mask = pd.DataFrame(index=df.index)

    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        mask = mask.fillna(False)
        outlier_mask[column] = mask

        outlier_count = int(mask.sum())
        outlier_percent = (
            outlier_count / total_rows * 100
            if total_rows > 0
            else 0
        )

        rows.append(
            {
                "feature": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
                "outlier_percent": outlier_percent,
            }
        )

    summary = (
        pd.DataFrame(rows)
        .set_index("feature")
        .sort_values("outlier_count", ascending=False)
    )

    if return_mask:
        return summary, outlier_mask

    return summary
