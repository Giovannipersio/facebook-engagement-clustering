"""
Reusable utilities for model tuning and evaluation.

These helpers keep notebook evaluation code concise while preserving a
consistent visual and tabular format across classifiers.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


__all__ = [
    "best_parameters_dataframe",
    "classification_report_dataframe",
    "evaluate_best_estimator",
    "plot_confusion_matrix",
    "summarize_model_performance",
    "tune_classifier",
]


def tune_classifier(
    estimator,
    param_grid,
    preprocessor,
    X_train,
    y_train,
    cv,
    scoring="accuracy",
    n_jobs=-1,
):
    """
    Tune a classifier inside a preprocessing pipeline.

    The preprocessor is included in the pipeline so transformations are learned
    only from the training split inside each cross-validation fold.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    return search


def best_parameters_dataframe(search):
    """Return the selected hyperparameters as a tidy dataframe."""
    best_params = {
        key.replace("classifier__", ""): value
        for key, value in search.best_params_.items()
    }

    return (
        pd.Series(best_params, name="Selected Value")
        .rename_axis("Hyperparameter")
        .reset_index()
    )


def summarize_model_performance(model_name, search, y_true, y_pred, digits=4):
    """Build a one-row summary with cross-validation and test metrics."""
    summary = pd.DataFrame(
        {
            "Model": [model_name],
            "Best CV Accuracy": [search.best_score_],
            "Test Accuracy": [accuracy_score(y_true, y_pred)],
            "Test Macro F1": [f1_score(y_true, y_pred, average="macro")],
            "Test Weighted F1": [f1_score(y_true, y_pred, average="weighted")],
        }
    )

    numeric_columns = summary.select_dtypes(include=["number"]).columns
    summary[numeric_columns] = summary[numeric_columns].round(digits)

    return summary


def classification_report_dataframe(y_true, y_pred, class_names, digits=4):
    """Return sklearn's classification report as a formatted dataframe."""
    report = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return pd.DataFrame(report).T.round(digits)


def evaluate_best_estimator(model_name, search, X_test, y_test, class_names):
    """Generate predictions, summary metrics, best parameters, and report."""
    y_pred = search.predict(X_test)

    return {
        "model_name": model_name,
        "predictions": y_pred,
        "summary": summarize_model_performance(
            model_name=model_name,
            search=search,
            y_true=y_test,
            y_pred=y_pred,
        ),
        "best_params": best_parameters_dataframe(search),
        "classification_report": classification_report_dataframe(
            y_true=y_test,
            y_pred=y_pred,
            class_names=class_names,
        ),
    }


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    model_name,
    figsize=(10, 8),
    cmap="Blues",
    normalize=None,
    colorbar=False,
):
    """
    Plot a standardized confusion matrix for multiclass classifiers.

    The labels assume the target classes were encoded as consecutive integers
    in the same order as class_names.
    """
    display_names = [
        str(class_name).replace("_", " ")
        for class_name in class_names
    ]
    labels = np.arange(len(display_names))
    values_format = ".2f" if normalize is not None else "d"

    fig, ax = plt.subplots(figsize=figsize)

    ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        display_labels=display_names,
        cmap=cmap,
        normalize=normalize,
        values_format=values_format,
        ax=ax,
        colorbar=colorbar,
    )

    ax.set_title(
        f"Confusion Matrix - {model_name}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Predicted Class", fontsize=12, labelpad=10)
    ax.set_ylabel("True Class", fontsize=12, labelpad=10)
    ax.grid(False)
    ax.tick_params(axis="y", rotation=0)

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    fig.tight_layout()
    plt.show()

    return fig, ax
