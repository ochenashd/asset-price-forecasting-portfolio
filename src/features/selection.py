"""
Feature selection utilities to reduce noise and improve model performance.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logger = logging.getLogger(__name__)


def _is_target_metadata(column: str) -> bool:
    """Reserve target fields, including numeric encodings of future-derived metadata."""
    return column == "target" or column.startswith("target_")


def remove_high_correlation(
    df: pd.DataFrame,
    threshold: float = 0.90,
    target_col: str = "target",
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """Remove one of highly correlated numeric features."""
    exclude_cols = exclude_cols or []
    protected_cols = set([target_col] + exclude_cols)

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [
        col for col in numeric if col not in protected_cols and not _is_target_metadata(col)
    ]

    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    selected = [col for col in feature_cols if col not in to_drop]
    logger.info(f"Correlation filter: dropped {len(to_drop)} features")
    return selected


def _is_classification_target(y: pd.Series, target_type: str = "5d_class") -> bool:
    """Choose the importance task explicitly and reject invalid class labels."""
    if target_type in {"5d_return", "1d_return"}:
        return False
    if target_type != "5d_class":
        raise ValueError(f"Unsupported target type: {target_type}")
    if (
        y.empty
        or not pd.api.types.is_numeric_dtype(y.dtype)
        or pd.api.types.is_bool_dtype(y.dtype)
        or pd.api.types.is_complex_dtype(y.dtype)
        or not y.isin([0, 1, 2]).all()
    ):
        raise ValueError("5d_class targets must contain only integer labels 0, 1, or 2.")
    return True


def select_by_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_select: int = 50,
    *,
    target_type: str = "5d_class",
    return_estimator: bool = False,
) -> List[str] | Tuple[List[str], object]:
    """Select features; optionally return the same fitted estimator, without data."""
    n_select = min(n_select, X.shape[1])
    if X.empty or n_select < 1 or any(_is_target_metadata(col) for col in X.columns):
        raise ValueError("Importance fitting requires non-empty predictors without target metadata.")

    if _is_classification_target(y, target_type):
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        )

    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns)
    selected = importance.nlargest(n_select).index.tolist()

    logger.info(f"Selected top {len(selected)} features by RF importance")
    return (selected, model) if return_estimator else selected


def run_feature_selection(
    df: pd.DataFrame,
    target_col: str = "target",
    n_features: int = 40,
    exclude_cols: Optional[List[str]] = None,
    keep_cols: Optional[List[str]] = None,
    *,
    target_type: str = "5d_class",
    return_selector: bool = False,
) -> Tuple[pd.DataFrame, List[str]] | Tuple[pd.DataFrame, List[str], object]:
    """Select fitting features; optional third result retains the actual forest.

    The default two-value contract and fitting algorithm are unchanged. The
    forest carries learned trees and feature metadata, not the training frame.
    """
    logger.info("Running feature selection...")

    exclude_cols = exclude_cols or []
    keep_cols = keep_cols or []

    selected_corr = remove_high_correlation(
        df,
        target_col=target_col,
        exclude_cols=exclude_cols + keep_cols,
    )
    X = df[selected_corr]
    y = df[target_col]

    if return_selector:
        final_features, selector = select_by_importance(X, y, n_select=n_features, target_type=target_type, return_estimator=True)
    else:
        final_features = select_by_importance(X, y, n_select=n_features, target_type=target_type)
    output_cols = final_features + [target_col] + [col for col in keep_cols if col in df.columns]
    final_df = df[output_cols].copy()

    return (final_df, final_features, selector) if return_selector else (final_df, final_features)
