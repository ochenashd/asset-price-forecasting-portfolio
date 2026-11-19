"""Verified, unfitted research inputs; no pipeline, estimator or trading calls.

V1 supports the pinned APF OHLCV recipe below, flat physical Parquet columns and
5d_class/5d_return targets. Learned selection/scaling must be explicitly deferred.
Unknown recipes are blockers, never approximated. Original labels remain source
metadata; feature clocks come exclusively from declared aware observations.
"""

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import logging
from pathlib import Path
import platform
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow

from src.data.preprocessor import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.evaluation.execution.prediction_export import _utc_instant
from .protocol import (
    ProtocolDiagnostic, ResearchProtocolError, ResearchProtocolV1, FrozenPartitionSpec,
    ProvenanceLimitations, ResearchArtifactRef, TransformationSpec, _digest, _encode, _key,
    read_verified_research_artifacts, validate_research_protocol,
)


def _implementation_reference():
    from src.data import preprocessor
    from src.features import engineering
    digest = hashlib.sha256()
    source_root = Path(__file__).resolve().parents[1]
    for path in (Path(__file__), Path(preprocessor.__file__), Path(engineering.__file__),
                 source_root / "features/selection.py", source_root / "models/ml.py"):
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return "apf-research-input-v1:" + digest.hexdigest()


def apf_research_input_recipe(feature_order, *, window_sizes=(7, 14, 30),
                              price_policy="causal_ffill", calendar_timezone="America/New_York",
                              raw_input_columns=("open", "high", "low", "close", "volume")):
    """Declare the exact supported recipe, with unfitted downstream stages.

    feature_order is a predeclared candidate subset in production generation
    order, not the output of a supervised selector. The adapter never fits the
    declared Random Forest or RobustScaler and never presents scaled features.
    """
    return TransformationSpec(
        {"column_policy": "exact_lowercase_ohlcv", "numeric_inputs": "float64",
         "feature_index": "declared_observation_instants", "other_columns": "raw_output_only"},
        "stable_source_order", "reject", {"price": price_policy, "volume": "preserve", "leading_missing": "preserve"},
        {"implementation": "APF-technical-and-calendar-v1", "window_sizes": tuple(window_sizes),
         "technical_indicators": True, "return_lags": (1, 2, 3, 5, 10, 20),
         "rsi_window": 14, "macd_spans": (12, 26, 9), "bollinger": (20, 2.0),
         "ewm_adjust": False, "std_ddof": 1, "volatility_shift": 1, "momentum_shift": 1,
         "volume_periods": (1, 7), "volume_shift": 1, "pct_change_fill": "library_default_pad",
         "pandas_version": pd.__version__, "feature_tail_policy": "prefix_only"},
        {"threshold": 0.90, "scope": "fitting", "stage": "deferred_fit"},
        {"method": "random_forest", "n_features": 40, "n_estimators": 300, "max_depth": 6,
         "classification_class_weight": "balanced", "random_state": 42, "n_jobs": -1,
         "scope": "fitting", "stage": "deferred_fit"},
        {"class": "RobustScaler", "quantile_range": (25.0, 75.0), "with_centering": True,
         "with_scaling": True, "unit_variance": False, "copy": True, "scope": "fitting", "stage": "deferred_fit"},
        tuple(raw_input_columns), tuple(feature_order), _implementation_reference(), calendar_timezone)


class ResearchInputError(ResearchProtocolError):
    """Structured fail-closed preparation error; no partial datasets are returned."""


def _block(code, field, message):
    raise ResearchInputError((ProtocolDiagnostic(code, field, message),))


@dataclass(frozen=True)
class ResearchRowBinding:
    row_id: object
    source_id: object
    observation_at: datetime
    target_end_source_id: object
    target_end_timestamp: datetime


@dataclass(frozen=True)
class FrozenResearchPartition:
    """Unscaled candidate inputs; public pandas properties return defensive copies."""
    row_bindings: tuple[ResearchRowBinding, ...]
    _features: pd.DataFrame = field(repr=False)
    _labels: pd.Series = field(repr=False)
    _target_returns: pd.Series = field(repr=False)

    @property
    def row_ids(self):
        return tuple(row.row_id for row in self.row_bindings)

    @property
    def features(self):
        return self._features.copy(deep=True)

    @property
    def labels(self):
        return self._labels.copy(deep=True)

    @property
    def target_returns(self):
        return self._target_returns.copy(deep=True)


@dataclass(frozen=True)
class PreparedResearchInputs:
    asset_id: str
    protocol_id: str
    source_training_input_id: str
    training_input_id: str
    source_artifacts: tuple[ResearchArtifactRef, ...]
    row_bindings: tuple[ResearchRowBinding, ...]
    frozen_partitions: FrozenPartitionSpec
    maturity_mapping: tuple
    fitting: FrozenResearchPartition
    inner_purged: FrozenResearchPartition
    validation: FrozenResearchPartition
    outer_purged: FrozenResearchPartition
    held_out: FrozenResearchPartition
    diagnostics: tuple[ProtocolDiagnostic, ...]
    provenance_limitations: ProvenanceLimitations
    _original: pd.DataFrame = field(repr=False)
    _cleaned: pd.DataFrame = field(repr=False)
    _engineered: pd.DataFrame = field(repr=False)
    selection_status: str = field(default="deferred_fitting_only", init=False)
    scaling_status: str = field(default="deferred_fitting_only", init=False)
    execution_eligible: bool = field(default=False, init=False)
    historical_point_in_time_verified: bool = field(default=False, init=False)

    @property
    def original_observations(self):
        """Verified physical columns with Arrow dtypes, never cleaned prices."""
        return self._original.copy(deep=True)

    @property
    def cleaned_observations(self):
        return self._cleaned.copy(deep=True)

    @property
    def engineered(self):
        return self._engineered.copy(deep=True)


def _check_recipe(protocol):
    recipe = protocol.transformations
    windows = recipe.indicators.get("window_sizes", ())
    columns = recipe.raw_input_columns
    if (not windows or any(type(value) is not int or value < 2 for value in windows)
            or len(set(windows)) != len(windows)
            or columns not in (("open", "high", "low", "close", "volume"),
                               ("open", "high", "low", "close", "volume", "adj_close"))
            or recipe.missing_values.get("price") not in {"causal_ffill", "preserve"}):
        _block("unsupported_recipe", "transformations", "V1 requires explicit APF windows and flat numeric OHLCV inputs.")
    expected = apf_research_input_recipe(recipe.feature_order, window_sizes=windows,
        price_policy=recipe.missing_values["price"], calendar_timezone=recipe.calendar_timezone,
        raw_input_columns=columns)
    if _encode(recipe) != _encode(expected):
        _block("unsupported_recipe", "transformations", "Effective normalization, indicators and deferred fitting stages must match the supported recipe exactly.")
    if protocol.target.task_type not in {"5d_class", "5d_return"} or protocol.target.price_column != "close":
        _block("unsupported_target", "target.task_type", "V1 supports the existing forward observation-return target paths.")
    required = {"pandas": pd.__version__, "numpy": np.__version__, "pyarrow": pyarrow.__version__}
    if (protocol.environment.python_version != platform.python_version()
            or any(protocol.environment.package_versions.get(name) != version for name, version in required.items())):
        _block("environment_mismatch", "environment.package_versions", "Decode and transformation library versions must match the frozen declaration.")


def _view_digest(partition):
    return _encode((partition.row_ids, tuple(partition._features.columns),
                    tuple(str(dtype) for dtype in partition._features.dtypes), str(partition._labels.dtype),
                    tuple(tuple(row) for row in partition._features.itertuples(index=False, name=None)),
                    tuple(partition._labels)))


def prepare_research_inputs(protocol: ResearchProtocolV1, research_root) -> PreparedResearchInputs:
    """Prepare exact frozen, unfitted partitions without reopening source paths.

    Artifact read images close before transformation; owned Arrow tables supply
    all consumed values. Physical clocks index model inputs; explicit immutable
    row bindings preserve opaque source and engineered identities separately.
    """
    try:
        validate_research_protocol(protocol).require_valid()
        _check_recipe(protocol)
        verified = read_verified_research_artifacts(protocol, research_root)
        primary = next(item for item in verified.artifacts if item.reference.role == "training_ohlcv")
        original = primary.to_pandas()
        if "asset_id" in original and any(not isinstance(value, str) or value != protocol.asset_id
                                          for value in original["asset_id"]):
            _block("asset_identity_mismatch", "asset_id", "An explicit source asset column must match the frozen asset exactly.")
        recipe, window = protocol.transformations, protocol.computation
        columns = recipe.raw_input_columns
        for name in columns:
            if not pd.api.types.is_numeric_dtype(original[name].dtype) or pd.api.types.is_bool_dtype(original[name].dtype):
                _block("unsupported_input_dtype", name, "Model normalization requires declared numeric OHLCV columns.")
        observations = protocol.universe.observations
        by_source = {_key(row.source_id): row for row in observations}
        raw_keys = tuple(_key(value) for value in original[primary.reference.row_id_column])
        if raw_keys != tuple(_key(row.source_id) for row in observations):
            _block("source_order_mismatch", "original_rows", "Decoded source keys differ from verified observations.")
        clocks = pd.DatetimeIndex([_utc_instant(by_source[key].observation_at) for key in raw_keys], name="observation_timestamp")
        clocks = clocks.tz_convert(recipe.calendar_timezone)
        model = original.loc[:, list(columns)].astype("float64").copy(deep=True)
        model.index = clocks
        if recipe.missing_values["price"] == "causal_ffill":
            # Avoid the preprocessor constructor's processed-directory creation.
            cleaned = DataPreprocessor.clean_data(SimpleNamespace(logger=logging.getLogger(__name__)), model)
        else:
            cleaned = model.copy(deep=True)
        if not cleaned.index.identical(model.index) or tuple(cleaned.columns) != columns:
            _block("normalization_changed_identity", "normalization", "The supported path must not sort or deduplicate original observations.")
        positions = {_key(row.source_id): i for i, row in enumerate(observations)}
        prefix_clocks = clocks[:positions[_key(window.feature_endpoint)] + 1]
        tail_clocks = clocks[:positions[_key(window.label_tail_endpoint)] + 1]
        target_config = {"type": protocol.target.task_type, "horizon": protocol.target.horizon}
        if protocol.target.task_type == "5d_class":
            target_config["threshold"] = protocol.target.threshold
        engineer = FeatureEngineer({"features": {"window_sizes": tuple(recipe.indicators["window_sizes"]), "technical_indicators": True},
                                   "target": target_config})
        features = engineer.add_calendar_features(engineer.add_technical_indicators(cleaned.loc[prefix_clocks].copy()))
        labels = engineer.add_target(cleaned.loc[tail_clocks].copy())
        features = features.drop(columns=["returns", "log_returns", "daily_return", "weekly_return"], errors="ignore")
        engineered = features.join(labels[["target", "target_return", "target_end_timestamp"]], how="left")
        engineered = engineered.replace([np.inf, -np.inf], np.nan).dropna().copy()
        by_clock = {_utc_instant(row.observation_at).isoformat(): row for row in observations}
        actual_sources = tuple(by_clock[_utc_instant(clock).isoformat()].source_id for clock in engineered.index)
        expected_sources = tuple(row.source_id for row in protocol.universe.engineered_rows)
        if tuple(map(_key, actual_sources)) != tuple(map(_key, expected_sources)):
            _block("engineered_universe_mismatch", "engineered_rows", "Actual feature exclusions or row order differ from the frozen universe.")
        natural_order = tuple(name for name in engineered.columns if name in recipe.feature_order)
        if natural_order != recipe.feature_order:
            _block("feature_order_mismatch", "feature_order", "Candidate features must exist in the frozen generation order.")
        mapping = {_key(row.source_id): row.row_id for row in protocol.universe.engineered_rows}
        maturity = {_key(row): end for row, end in protocol.target.maturity}
        bindings = []
        for clock, source_id in zip(engineered.index, actual_sources):
            row_id = mapping[_key(source_id)]
            end_id = maturity[_key(row_id)]
            end_clock = by_source[_key(end_id)].observation_at
            if _utc_instant(engineered.loc[clock, "target_end_timestamp"]) != _utc_instant(end_clock):
                _block("target_maturity_mismatch", "target.maturity", "Derived labels must retain the original unfiltered maturity relationship.")
            bindings.append(ResearchRowBinding(row_id, source_id, clock, end_id, end_clock))
        by_row = {_key(row.row_id): row for row in bindings}
        def partition(ids):
            selected = tuple(by_row[_key(row_id)] for row_id in ids)
            index = pd.DatetimeIndex([row.observation_at for row in selected], tz=clocks.tz, name=clocks.name)
            return FrozenResearchPartition(selected, engineered.loc[index, list(recipe.feature_order)].copy(deep=True),
                                           engineered.loc[index, "target"].copy(deep=True), engineered.loc[index, "target_return"].copy(deep=True))
        p = protocol.partitions
        fitting, inner, validation, outer, held_out = (partition(ids) for ids in (p.fitting, p.inner_purge, p.validation, p.outer_purge, p.held_out))
        if p.validation:
            cutoff = by_row[_key(p.validation_start)].observation_at
            if any(_utc_instant(row.target_end_timestamp) >= _utc_instant(cutoff) for row in fitting.row_bindings):
                _block("inner_maturity_violation", "fitting", "Fitting labels must mature strictly before validation.")
        test_cutoff = by_row[_key(p.test_start)].observation_at
        if any(_utc_instant(row.target_end_timestamp) >= _utc_instant(test_cutoff) for row in fitting.row_bindings + validation.row_bindings):
            _block("outer_maturity_violation", "training", "Fitting/validation labels must mature strictly before held-out observations.")
        training_id = "apf-prepared-training-v1-" + _digest((protocol.training_input_id, _view_digest(fitting), _view_digest(validation)))
        filled = int((model.isna() & cleaned.notna()).sum().sum())
        diagnostics = (ProtocolDiagnostic("verified_read_images", "artifacts", "All consumed data came from verified private read images."),
                       ProtocolDiagnostic("asset_identity_binding", "asset_id", "Asset identity is protocol-declared; a source asset_id column, when present, is checked. No independent instrument authentication is claimed."),
                       ProtocolDiagnostic("causal_fill_count", "normalization", str(filled)),
                       ProtocolDiagnostic("deferred_fitting_stages", "transformations", "Correlation, supervised selection and scaling remain unfitted and fitting-only."))
        return PreparedResearchInputs(protocol.asset_id, protocol.protocol_id, protocol.training_input_id, training_id,
            tuple(item.reference for item in verified.artifacts), tuple(bindings), p, protocol.target.maturity,
            fitting, inner, validation, outer, held_out, diagnostics, protocol.provenance,
            original.copy(deep=True), cleaned.copy(deep=True), engineered.copy(deep=True))
    except ResearchInputError:
        raise
    except ResearchProtocolError as exc:
        raise ResearchInputError(exc.diagnostics) from exc
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OSError) as exc:
        _block("input_preparation_failed", "inputs", type(exc).__name__)
