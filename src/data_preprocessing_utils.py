"""Reusable utilities for preprocessing tabular data.

The helpers in this module are domain-independent and designed to keep data
science notebooks focused on project decisions instead of implementation
details.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd


if TYPE_CHECKING:
    from sklearn.compose import ColumnTransformer


__all__ = [
    "add_cyclical_features",
    "add_datetime_features",
    "build_clustering_preprocessor",
    "component_consistency_summary",
    "validate_nonnegative_columns",
    "validate_numeric_matrix",
]


def _require_columns(
    dataframe: pd.DataFrame,
    columns: Sequence[Any],
) -> None:
    """Raise a clear error when requested columns are unavailable."""
    missing_columns = set(columns).difference(dataframe.columns)

    if missing_columns:
        ordered_missing_columns = sorted(missing_columns, key=str)
        raise ValueError(
            f"Required columns are missing: {ordered_missing_columns}."
        )


def validate_nonnegative_columns(
    dataframe: pd.DataFrame,
    columns: Sequence[Any],
) -> pd.Series:
    """Validate that selected numeric columns contain no negative values.

    Returns a series with the number of negative values in each column. The
    returned counts are useful when building notebook quality summaries.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    selected_columns = list(columns)
    _require_columns(dataframe, selected_columns)

    nonnumeric_columns = [
        column
        for column in selected_columns
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    if nonnumeric_columns:
        raise TypeError(
            f"Columns must be numeric: {sorted(nonnumeric_columns, key=str)}."
        )

    negative_counts = dataframe[selected_columns].lt(0).sum()

    if negative_counts.any():
        invalid_counts = negative_counts[negative_counts > 0].to_dict()
        raise ValueError(f"Negative values found: {invalid_counts}.")

    return negative_counts


def add_datetime_features(
    dataframe: pd.DataFrame,
    source_column: Any,
    *,
    datetime_format: str | None = None,
    parsed_column: str = "parsed_datetime",
    feature_prefix: str = "datetime",
    allow_missing: bool = False,
) -> pd.DataFrame:
    """Parse a datetime column and add common calendar features.

    The returned dataframe is a copy. The original source column is preserved,
    and the generated hour includes minutes as a fractional value.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    _require_columns(dataframe, [source_column])

    result = dataframe.copy(deep=True)
    parsed_values = pd.to_datetime(
        result[source_column],
        format=datetime_format,
        errors="coerce",
    )

    invalid_count = int(parsed_values.isna().sum())

    if invalid_count and not allow_missing:
        raise ValueError(
            f"Column {source_column!r} contains {invalid_count} invalid or "
            "missing datetime values."
        )

    result[parsed_column] = parsed_values
    result[f"{feature_prefix}_year"] = parsed_values.dt.year
    result[f"{feature_prefix}_month"] = parsed_values.dt.month
    result[f"{feature_prefix}_weekday"] = parsed_values.dt.day_name()
    result[f"{feature_prefix}_weekday_number"] = parsed_values.dt.dayofweek
    result[f"{feature_prefix}_hour"] = (
        parsed_values.dt.hour + parsed_values.dt.minute / 60
    )

    return result


def add_cyclical_features(
    dataframe: pd.DataFrame,
    feature_periods: Mapping[Any, float],
    *,
    offsets: Mapping[Any, float] | None = None,
) -> pd.DataFrame:
    """Add sine and cosine encodings for one or more cyclical columns.

    Parameters
    ----------
    dataframe
        Source dataframe.
    feature_periods
        Mapping from each source column to its cycle length.
    offsets
        Optional values subtracted before encoding. For example, use an offset
        of 1 for calendar months represented from 1 through 12.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    if not feature_periods:
        raise ValueError("feature_periods must contain at least one feature.")

    _require_columns(dataframe, list(feature_periods))
    offsets = {} if offsets is None else offsets
    result = dataframe.copy(deep=True)

    for column, period in feature_periods.items():
        if period <= 0:
            raise ValueError(
                f"The period for column {column!r} must be greater than zero."
            )

        values = pd.to_numeric(result[column], errors="raise")

        if values.isna().any():
            raise ValueError(
                f"Column {column!r} contains missing values and cannot be "
                "encoded cyclically."
            )

        adjusted_values = values - offsets.get(column, 0.0)
        angles = 2 * np.pi * adjusted_values / period
        result[f"{column}_sin"] = np.sin(angles)
        result[f"{column}_cos"] = np.cos(angles)

    return result


def component_consistency_summary(
    dataframe: pd.DataFrame,
    total_column: Any,
    component_columns: Sequence[Any],
) -> pd.Series:
    """Compare a reported total with the row-wise sum of its components."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    components = list(component_columns)
    _require_columns(dataframe, [total_column, *components])

    columns_to_check = [total_column, *components]
    nonnumeric_columns = [
        column
        for column in columns_to_check
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    if nonnumeric_columns:
        raise TypeError(
            f"Columns must be numeric: {sorted(nonnumeric_columns, key=str)}."
        )

    difference = dataframe[total_column] - dataframe[components].sum(axis=1)

    return pd.Series(
        {
            "exact_matches": int(difference.eq(0).sum()),
            "different_rows": int(difference.ne(0).sum()),
            "maximum_absolute_difference": float(difference.abs().max()),
        },
        name="value",
    )


def build_clustering_preprocessor(
    *,
    log_scaled_columns: Sequence[Any] = (),
    standard_scaled_columns: Sequence[Any] = (),
    categorical_columns: Sequence[Any] = (),
) -> ColumnTransformer:
    """Build a pandas-output preprocessor for distance-based clustering.

    Non-negative skewed features receive ``log1p`` followed by standard
    scaling. Other continuous features receive standard scaling, while
    categorical features receive dense one-hot encoding.
    """
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import (
            FunctionTransformer,
            OneHotEncoder,
            StandardScaler,
        )
    except ImportError as error:
        raise ImportError(
            "build_clustering_preprocessor requires scikit-learn. "
            "Install it with 'pip install scikit-learn'."
        ) from error

    log_columns = list(log_scaled_columns)
    standard_columns = list(standard_scaled_columns)
    category_columns = list(categorical_columns)
    configured_columns = [log_columns, standard_columns, category_columns]

    flattened_columns = [
        column for column_group in configured_columns for column in column_group
    ]

    if not flattened_columns:
        raise ValueError("At least one preprocessing column must be provided.")

    duplicated_columns = {
        column
        for column in flattened_columns
        if flattened_columns.count(column) > 1
    }

    if duplicated_columns:
        raise ValueError(
            "Columns cannot belong to multiple preprocessing groups: "
            f"{sorted(duplicated_columns, key=str)}."
        )

    transformers: list[tuple[str, Any, list[Any]]] = []

    if log_columns:
        log_pipeline = Pipeline(
            steps=[
                (
                    "log_transform",
                    FunctionTransformer(
                        np.log1p,
                        feature_names_out="one-to-one",
                    ),
                ),
                ("standard_scaler", StandardScaler()),
            ]
        )
        transformers.append(("log_scaled", log_pipeline, log_columns))

    if standard_columns:
        standard_pipeline = Pipeline(
            steps=[("standard_scaler", StandardScaler())]
        )
        transformers.append(
            ("standard_scaled", standard_pipeline, standard_columns)
        )

    if category_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "one_hot_encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                )
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, category_columns)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")


def validate_numeric_matrix(
    dataframe: pd.DataFrame,
    *,
    excluded_columns: Sequence[Any] = (),
    expected_row_count: int | None = None,
) -> pd.DataFrame:
    """Validate and return the numeric feature portion of a model matrix."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    excluded = list(excluded_columns)
    _require_columns(dataframe, excluded)

    if expected_row_count is not None and len(dataframe) != expected_row_count:
        raise ValueError(
            f"Expected {expected_row_count} rows, but found {len(dataframe)}."
        )

    if not dataframe.columns.is_unique:
        raise ValueError("The model matrix contains duplicated columns.")

    feature_values = dataframe.drop(columns=excluded)

    if feature_values.shape[1] == 0:
        raise ValueError("The model matrix does not contain feature columns.")

    if feature_values.isna().any().any():
        raise ValueError("The model matrix contains missing values.")

    nonnumeric_columns = [
        column
        for column in feature_values
        if not pd.api.types.is_numeric_dtype(feature_values[column])
    ]

    if nonnumeric_columns:
        raise TypeError(
            "The model matrix contains non-numeric columns: "
            f"{sorted(nonnumeric_columns, key=str)}."
        )

    if not np.isfinite(feature_values.to_numpy()).all():
        raise ValueError("The model matrix contains infinite values.")

    return feature_values
