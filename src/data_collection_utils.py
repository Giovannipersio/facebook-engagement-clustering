"""Reusable utilities for collecting and validating tabular datasets.

The functions in this module are intentionally domain-independent so they can
be reused across notebook-based data science projects.
"""

from collections.abc import Collection
from pathlib import Path
from typing import Any

import pandas as pd


__all__ = [
    "dataframe_overview",
    "fetch_uci_features",
    "save_dataframe_csv",
    "validate_dataframe_contract",
]


def fetch_uci_features(dataset_id: int) -> pd.DataFrame:
    """Download and return a copy of a UCI dataset's feature table.

    Parameters
    ----------
    dataset_id
        Numeric identifier shown on the UCI Machine Learning Repository page.

    Returns
    -------
    pandas.DataFrame
        An independent copy of the dataset's feature table.

    Raises
    ------
    TypeError
        If ``dataset_id`` is not an integer.
    ValueError
        If ``dataset_id`` is not positive or the dataset has no feature table.
    """
    if isinstance(dataset_id, bool) or not isinstance(dataset_id, int):
        raise TypeError("dataset_id must be an integer.")

    if dataset_id <= 0:
        raise ValueError("dataset_id must be greater than zero.")

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as error:
        raise ImportError(
            "fetch_uci_features requires the optional 'ucimlrepo' package. "
            "Install it with 'pip install ucimlrepo'."
        ) from error

    dataset = fetch_ucirepo(id=dataset_id)
    features = dataset.data.features

    if features is None:
        raise ValueError(
            f"UCI dataset {dataset_id} does not contain a feature table."
        )

    if not isinstance(features, pd.DataFrame):
        raise TypeError("The UCI feature table is not a pandas DataFrame.")

    return features.copy(deep=True)


def validate_dataframe_contract(
    dataframe: pd.DataFrame,
    *,
    required_columns: Collection[Any] | None = None,
    allow_empty: bool = False,
    require_unique_columns: bool = True,
) -> None:
    """Validate general structural requirements for a dataframe.

    Parameters
    ----------
    dataframe
        Dataframe to validate.
    required_columns
        Column labels that must be present. Extra columns are allowed.
    allow_empty
        Whether a dataframe with no rows is considered valid.
    require_unique_columns
        Whether duplicated column labels should raise an error.

    Raises
    ------
    TypeError
        If ``dataframe`` is not a pandas DataFrame.
    ValueError
        If one or more requested structural requirements are not met.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    if not allow_empty and dataframe.empty:
        raise ValueError("The dataframe is empty.")

    if require_unique_columns and not dataframe.columns.is_unique:
        duplicated_columns = (
            dataframe.columns[dataframe.columns.duplicated()].unique().tolist()
        )
        raise ValueError(
            f"The dataframe contains duplicated columns: {duplicated_columns}."
        )

    if required_columns is None:
        return

    missing_columns = set(required_columns).difference(dataframe.columns)

    if missing_columns:
        ordered_missing_columns = sorted(missing_columns, key=str)
        raise ValueError(
            f"Required columns are missing: {ordered_missing_columns}."
        )


def save_dataframe_csv(
    dataframe: pd.DataFrame,
    output_path: str | Path,
    *,
    index: bool = False,
    create_parent_directories: bool = True,
    **to_csv_kwargs: Any,
) -> Path:
    """Save a dataframe as CSV and return the resolved output path.

    Parameters
    ----------
    dataframe
        Dataframe to save.
    output_path
        Destination path, including the ``.csv`` file name.
    index
        Whether to write the dataframe index.
    create_parent_directories
        Whether missing parent directories should be created automatically.
    **to_csv_kwargs
        Additional keyword arguments forwarded to ``DataFrame.to_csv``.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    path = Path(output_path).expanduser().resolve()

    if path.suffix.lower() != ".csv":
        raise ValueError("output_path must use the .csv extension.")

    if create_parent_directories:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.parent.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {path.parent}"
        )

    dataframe.to_csv(path, index=index, **to_csv_kwargs)
    return path


def dataframe_overview(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a column-level structural and completeness summary."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    overview = pd.DataFrame(
        {
            "dtype": dataframe.dtypes.astype(str),
            "non_null": dataframe.notna().sum(),
            "missing": dataframe.isna().sum(),
            "missing_pct": dataframe.isna().mean().mul(100),
            "unique": dataframe.nunique(dropna=True),
        }
    )

    overview.index.name = "column"
    return overview
