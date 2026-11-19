# src/models/ml.py
"""
Machine learning models with feature scaling and temporal validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from dataclasses import dataclass
from omegaconf import DictConfig
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from src.features.engineering import validate_target_end_timestamps

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemporalSplit:
    """Original row identities and the mature fitting/validation subsets."""

    source_index: pd.DatetimeIndex
    fitting_index: pd.DatetimeIndex
    validation_index: pd.DatetimeIndex
    purged_count: int


def make_temporal_split(index, target_end_timestamps, training_config, validation_start_timestamp=None):
    """Choose and purge one split before any supervised feature selection."""
    validate_target_end_timestamps(index, target_end_timestamps)
    val_size = int(len(index) * float(training_config.get("validation_size", 0.2)))
    min_val = int(training_config.get("min_validation_samples", 60))
    use_validation = val_size > 0 and val_size >= min_val and len(index) - val_size >= min_val
    if validation_start_timestamp is not None:
        if validation_start_timestamp not in index:
            raise ValueError("The fixed validation boundary must be present in the input index.")
        val_size = int((index >= validation_start_timestamp).sum())
        use_validation = True
    validation_index = index[-val_size:] if use_validation else index[:0]
    candidates = index[:-val_size] if use_validation else index
    if use_validation:
        mature = target_end_timestamps.loc[candidates] < validation_index[0]
        fitting_index = candidates[mature]
    else:
        fitting_index = candidates
    if fitting_index.empty:
        raise ValueError("No mature fitting labels remain before validation.")
    return TemporalSplit(
        index.copy(), fitting_index.copy(), validation_index.copy(),
        len(candidates) - len(fitting_index),
    )


class MLModel:
    """Base class for ML models with built-in scaling."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.model_type = None
        self.feature_names = None
        self.ticker = None
        # Runtime-only tracking policy; never an estimator parameter or fitted state.
        self.legacy_mlflow_enabled = True

        scaler_type = config.get("features", {}).get("scaler", "robust")
        self.scaler_class = StandardScaler if scaler_type == "standard" else RobustScaler

    def fit(
        self, X: pd.DataFrame, y: pd.Series, ticker: str, *,
        target_end_timestamps: pd.Series | None = None,
        validation_start_timestamp: pd.Timestamp | None = None,
        temporal_split: TemporalSplit | None = None,
    ):
        """Purge labels before scaling and fitting, keeping validation rows fixed.

        Maturity must be supplied on the original observation index. Callers must
        enforce their outer fitting cutoff, including when validation is disabled.
        An explicit validation start preserves a boundary chosen before outer purging.
        A precomputed split is reused without splitting the fitting subset again.
        """
        validate_target_end_timestamps(X.index, target_end_timestamps)
        if X.empty or not y.index.equals(X.index):
            raise ValueError("X and y require non-empty, exactly aligned chronological observations.")
        if "target_end_timestamp" in X.columns:
            raise ValueError("target_end_timestamp is metadata and must not be a predictor.")
        self.ticker = ticker
        self.feature_names = X.columns.tolist()

        logger.info(
            f"Fitting {self.model_type} on {ticker} | Samples: {len(X)} | Features: {X.shape[1]}"
        )

        split = temporal_split
        if split is None:
            split = make_temporal_split(
                X.index, target_end_timestamps, self.config.get("training", {}),
                validation_start_timestamp,
            )
        if not split.source_index.equals(X.index):
            raise ValueError("The precomputed split must match the unchanged input index.")
        if (
            split.fitting_index.empty
            or not split.fitting_index.isin(X.index).all()
            or not split.validation_index.isin(X.index).all()
            or not split.fitting_index.intersection(split.validation_index).empty
        ):
            raise ValueError("Invalid precomputed fitting/validation indices.")
        X_train = X.loc[split.fitting_index].copy()
        y_train = y.loc[split.fitting_index].copy()
        self.training_purged_count = split.purged_count
        if not split.validation_index.empty:
            cutoff = split.validation_index[0]
            expected_validation = X.index[X.index >= cutoff]
            expected_fitting = X.index[
                (X.index < cutoff) & (target_end_timestamps < cutoff)
            ]
            if (
                not split.validation_index.equals(expected_validation)
                or not split.fitting_index.equals(expected_fitting)
                or (validation_start_timestamp is not None and cutoff != validation_start_timestamp)
            ):
                raise ValueError("Precomputed split violates the fixed boundary or label maturity.")
            X_val = X.loc[split.validation_index].copy()
            y_val = y.loc[split.validation_index].copy()
            logger.info(
                "Purged %d fitting labels at validation cutoff %s",
                self.training_purged_count, X_val.index[0],
            )
        else:
            if not split.fitting_index.equals(X.index) or validation_start_timestamp is not None:
                raise ValueError("A split without validation must retain all permitted input rows.")
            X_val = None
            y_val = None

        self.scaler = self.scaler_class()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )

        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = pd.DataFrame(
                self.scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index,
            )

        self._fit_model(X_train_scaled, y_train, X_val_scaled, y_val)

        logger.info(f"{self.model_type} training completed with {self.scaler_class.__name__}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with scaling applied."""
        if self.model is None:
            raise ValueError("Model not fitted yet")

        if self.scaler is None:
            logger.warning("Scaler not fitted - predicting without scaling")
            return self._predict_model(X)

        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index,
        )
        return self._predict_model(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities when the underlying model supports it."""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(f"{self.model_type} does not support predict_proba")

        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index,
        )
        return self.model.predict_proba(X_scaled)

    def _fit_model(self, X_scaled: pd.DataFrame, y: pd.Series, X_val_scaled=None, y_val=None):
        """To be implemented by child classes."""
        raise NotImplementedError

    def _predict_model(self, X_scaled: pd.DataFrame) -> np.ndarray:
        """To be implemented by child classes."""
        raise NotImplementedError

    def save(self, path: Path):
        """Save the fitted model and scaler."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "model_type": self.model_type,
            },
            path,
        )
        logger.info(f"{self.model_type} model + scaler saved: {path}")


class XGBoostModel(MLModel):
    """XGBoost model for regression or three-class direction classification."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.model_type = "XGBoost"
        self.params = config.get("models", {}).get("xgboost", {})
        self.task = self.params.get("task", config.get("target", {}).get("type", "5d_class"))

        self.default_classifier_params = {
            "n_estimators": 1000,
            "learning_rate": 0.02,
            "max_depth": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "random_state": 42,
            "early_stopping_rounds": 100,
        }

        self.default_regressor_params = {
            "n_estimators": 1000,
            "learning_rate": 0.02,
            "max_depth": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "random_state": 42,
            "early_stopping_rounds": 100,
        }

    def _is_classification(self, y: pd.Series) -> bool:
        """Return True when the target contains the down/neutral/up classes."""
        unique_values = set(pd.Series(y).dropna().unique().tolist())
        return unique_values.issubset({0, 1, 2}) and len(unique_values) <= 3

    def _fit_model(self, X_scaled: pd.DataFrame, y: pd.Series, X_val_scaled=None, y_val=None):
        """Fit XGBoost using a proper temporal validation set when available."""
        import xgboost as xgb
        legacy_tracking = getattr(self, "legacy_mlflow_enabled", True) is True
        if legacy_tracking:
            import mlflow

        is_classification = self._is_classification(y)
        base_params = self.default_classifier_params if is_classification else self.default_regressor_params
        params = {**base_params, **dict(self.params)}
        params.pop("task", None)

        eval_set = None
        if X_val_scaled is not None and y_val is not None:
            eval_set = [(X_val_scaled, y_val)]
        else:
            params.pop("early_stopping_rounds", None)

        sample_weight = self._build_sample_weight(y)

        if is_classification:
            self.model = xgb.XGBClassifier(**params)
            if legacy_tracking and mlflow.active_run() is not None:
                actual_params = self.model.get_params()

                for key, value in actual_params.items():
                    mlflow.log_param(f"xgboost_actual.{key}", value)
            self.model.fit(X_scaled, y.astype(int), sample_weight=sample_weight, eval_set=eval_set, verbose=False)
        else:
            self.model = xgb.XGBRegressor(**params)
            if legacy_tracking and mlflow.active_run() is not None:
                actual_params = self.model.get_params()

                for key, value in actual_params.items():
                    mlflow.log_param(f"xgboost_actual.{key}", value)
            self.model.fit(X_scaled, y, eval_set=eval_set, sample_weight=sample_weight, verbose=False)

        if hasattr(self.model, "best_iteration") and self.model.best_iteration is not None:
            logger.info(f"{self.model_type} best iteration: {self.model.best_iteration}")

    def _predict_model(self, X_scaled: pd.DataFrame) -> np.ndarray:
        """Return class labels for classification or numeric predictions for regression."""
        return self.model.predict(X_scaled)

    def _build_sample_weight(self, y_train: pd.Series) -> np.ndarray | None:
        """Build sample weights for multiclass classification."""
        from sklearn.utils.class_weight import compute_sample_weight
        import numpy as np

        class_weight_mode = self.params.get("class_weight_mode", "none")

        if class_weight_mode == "none":
            return None

        if class_weight_mode == "balanced":
            return compute_sample_weight(
                class_weight="balanced",
                y=y_train,
            )

        if class_weight_mode == "sqrt_balanced":
            balanced_weight = compute_sample_weight(
                class_weight="balanced",
                y=y_train,
            )
            return np.sqrt(balanced_weight)

        if class_weight_mode == "custom":
            class_weights = self.params.get(
                "class_weights",
                {
                    0: 1.0,
                    1: 1.0,
                    2: 1.0,
                },
            )

            return y_train.map(class_weights).astype(float).values

        raise ValueError(f"Unknown class_weight_mode: {class_weight_mode}")


class LightGBMModel(MLModel):
    """LightGBM model for regression."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.model_type = "LightGBM"
        self.params = config.get("models", {}).get("lightgbm", {})

        self.default_params = {
            "n_estimators": 800,
            "learning_rate": 0.03,
            "max_depth": 7,
            "num_leaves": 64,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
            "min_child_samples": 20,
            "reg_lambda": 2.0,
            "reg_alpha": 0.1,
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "random_state": 42,
        }

    def _fit_model(self, X_scaled: pd.DataFrame, y: pd.Series, X_val_scaled=None, y_val=None):
        """Fit LightGBM."""
        import lightgbm as lgb

        params = {**self.default_params, **dict(self.params)}
        self.model = lgb.LGBMRegressor(**params)

        eval_set = [(X_val_scaled, y_val)] if X_val_scaled is not None and y_val is not None else None
        self.model.fit(X_scaled, y, eval_set=eval_set)

    def _predict_model(self, X_scaled: pd.DataFrame) -> np.ndarray:
        """Predict numeric values."""
        return self.model.predict(X_scaled)


def get_ml_model(model_name: str, config: DictConfig) -> MLModel:
    """Create an ML model by name."""
    model_name = model_name.upper()
    if model_name in ["XGBOOST", "XGB"]:
        return XGBoostModel(config)
    if model_name in ["LIGHTGBM", "LGBM"]:
        return LightGBMModel(config)
    raise ValueError(f"Unknown ML model: {model_name}")


if __name__ == "__main__":
    print("ML models module loaded successfully")
