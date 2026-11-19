"""Offline synthetic research. No provider certification or investment claims.

The calendar is an artificial weekday calendar with explicit UTC clocks, not an
exchange calendar. Execution prices are a separately generated synthetic feed.
Only this run's freshly generated model bundle is loaded for functional replay.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
from importlib import metadata, resources
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from src.features.selection import run_feature_selection
from src.research import protocol as contracts
from src.research.input_adapter import apf_research_input_recipe, prepare_research_inputs
from src.research.receipt import ResearchFitAttempt, inspect_research_fit, create_research_fit_receipt
from src.research.bundle_preparation import capture_research_bundle, finalize_research_bundle
from src.research.bundle_publication import ResearchPublicationContext, RESEARCH_ARCHIVE_POLICY, publish_prepared_research_bundle
from src.research.functional_replay import replay_research_bundle
from src.evaluation.execution.adapter import AssetSpec
from src.evaluation.execution.contracts import ExecutionConfig, MarketBar, Signal
from src.evaluation.execution.engine import ExecutionEngineV1
from src.evaluation.execution.metrics import PerformanceConfig, calculate_performance
from src.evaluation.execution.model_provenance import SourceRow, FittedStateAvailability, FrozenModelManifest
from src.evaluation.execution.pipeline_handoff import bind_fresh_capture
from src.evaluation.execution.xgboost_bridge import capture_xgboost_predictions

UTC = timezone.utc
FEATURES = ("close", "return_lag_1", "ema_7", "vol_30", "volume_change", "dayofweek")
OHLCV = ("open", "high", "low", "close", "volume")
PARAMETERS = dict(n_estimators=24, max_depth=2, learning_rate=0.1, n_jobs=1,
                  random_state=42, tree_method="hist", objective="multi:softprob",
                  num_class=3, eval_metric="mlogloss", early_stopping_rounds=4)


def generate_ohlcv(sessions=180, seed=42):
    """Return independent owned observations on explicit synthetic UTC sessions."""
    if sessions < 180:
        raise ValueError("At least 180 sessions are required by the frozen v1 split.")
    t = np.arange(sessions)
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(.01 * np.sin(.41*t) + .003*np.sin(1.13*t) + .0002*t)
    opening = close * (1 + .001*np.sin(.23*t))
    days = pd.bdate_range("2024-01-01", periods=sessions, tz="UTC")
    # Execution observations have their own random stream and no model targets.
    independent_close = 100 * np.exp(np.cumsum(rng.normal(0, .003, sessions)))
    independent_open = np.r_[100., independent_close[:-1]] * np.exp(np.random.default_rng(seed+1).normal(0, .001, sessions))
    frame = pd.DataFrame(dict(source_id=[f"r{i}" for i in t], date_label=[d.date() for d in days],
        open=opening, high=np.maximum(opening, close)*1.003, low=np.minimum(opening, close)*.997,
        close=close, volume=1000+t%17))
    prices = pd.DataFrame(dict(session_id=[f"session-{i}" for i in t],
        open_at=[d.to_pydatetime()+timedelta(hours=9, minutes=30) for d in days],
        close_at=[d.to_pydatetime()+timedelta(hours=16) for d in days],
        open=independent_open, close=independent_close))
    return frame, prices


@contextmanager
def offline_guard(root):
    """Reject networking and SQLite outside the newly owned output root."""
    active = {"value": True}
    def audit(name, args):
        if not active["value"]:
            return
        if name in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
            raise RuntimeError("Network access is forbidden during the synthetic demonstration.")
        if name == "sqlite3.connect":
            target = Path(os.fsdecode(args[0])).resolve()
            if root not in target.parents:
                raise RuntimeError("Only the newly created local backend is permitted.")
    sys.addaudithook(audit)
    previous = {key: os.environ.get(key) for key in ("MLFLOW_DISABLE_TELEMETRY", "MLFLOW_ENABLE_WORKSPACES")}
    os.environ.update(MLFLOW_DISABLE_TELEMETRY="true", MLFLOW_ENABLE_WORKSPACES="false")
    try:
        yield
    finally:
        active["value"] = False
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_protocol(root, frame, prices):
    """Freeze original memberships before supervised selection or estimator fitting."""
    source = root/"inputs/prices.parquet"
    source.parent.mkdir()
    frame.iloc[:180].to_parquet(source, index=False)
    frame = frame.iloc[:180]
    schema = pq.read_schema(source)
    sha = lambda data: hashlib.sha256(data).hexdigest()
    ids = tuple(frame.source_id)
    clocks = tuple(prices.close_at.iloc[:180])
    e = lambda start, stop: tuple(f"e{i}" for i in range(start, stop))
    ref = contracts.ResearchArtifactRef("inputs/prices.parquet", "synthetic-prices", "training_ohlcv",
        sha(source.read_bytes()), source.stat().st_size,
        tuple(contracts.ColumnSpec(f.name, str(f.type), f.nullable) for f in schema),
        sha(schema.serialize().to_pybytes()), "source_id", ids, label_column="date_label")
    universe = contracts.ObservationUniverse(tuple(contracts.SourceObservation(key, ref.artifact_id,
        frame.date_label.iloc[i], "date32[day]", clocks[i], f"session-{i}", i) for i,key in enumerate(ids)),
        tuple(contracts.EngineeredRow(f"e{i}", f"r{i}") for i in range(31,175)),
        tuple((key, "warm_up" if i<31 else "label_tail") for i,key in enumerate(ids) if i<31 or i>=175))
    values = pa.Table.from_pandas(frame, preserve_index=False).to_pylist()[:120]
    profile = yaml.safe_load(resources.files("src").joinpath("public_defaults/environment.yaml").read_text())
    return contracts.ResearchProtocolV1("synthetic:SYNTH", "Public synthetic research v1", (ref,), universe,
        contracts.ComputationWindow(ids[:175], ids[174], ids[179], ids[120], ids[179]),
        apf_research_input_recipe(FEATURES, calendar_timezone="UTC"),
        contracts.TargetSpec("5d_class", 5, "observations", tuple((f"e{i}", f"r{i+5}") for i in range(31,175))),
        contracts.FrozenPartitionSpec(e(31,91),e(91,96),e(96,115),e(115,120),e(120,175),"e31","e96","e120"),
        contracts.EvaluationSpec("e120","e174",prices.open_at.iloc[120],clocks[179],"session-179",
            "liquidate_at_final_close",5,"sessions",True,("Synthetic daily close marks; no intraday risk.",),
            dict(position_policy="fixed_horizon",initial_capital=10000.,allocation_fraction=.5,
                 fee_rate=.001,slippage_rate=.0005,allow_short=False,
                 initial_margin_fraction=.5,maintenance_margin_fraction=.3,max_short_exposure=1.),
            dict(periods_per_year=252,annual_risk_free_rate=.02,calendar_days_per_year=365.25),
            calendar_reference="synthetic-weekday-utc-v1"),
        contracts.LearningSpec("xgboost",PARAMETERS,dict(estimator=42,selector=42),"fitting_only",
            "fixed_purged_validation",dict(enabled=True,rounds=4,information_scope="validation"),PARAMETERS),
        contracts.EnvironmentSpec("apf-public-v1",False,"reject",profile.get("source_digest"),platform.python_version(),
            {name:metadata.version(name) for name in ("numpy","pandas","pyarrow","scikit-learn","xgboost","scipy")},
            None,dict(architecture=platform.machine(),xgboost_workers=1,blas_threads="not_enforced")),
        contracts.ProvenanceLimitations("declared_pending_verification","unknown",
            ("Artificial UTC weekday calendar and independently generated synthetic execution feed.",),
            ("historical_price_vintage","exchange_calendar","provider_authentication"),evidence_origin="synthetic_fixture"),
        contracts.TrainingInputSpec(ids[:120], contracts.hash_training_rows(
            (row["source_id"],tuple(row[name] for name in OHLCV)) for row in values)))


def fit_states(prepared):
    """Fit selection and scaling only on the mature fitting subset."""
    fitting = prepared.fitting.features
    fitting["target"] = prepared.fitting.labels
    _, names, selector = run_feature_selection(fitting,n_features=40,target_type="5d_class",return_selector=True)
    scaler = RobustScaler().fit(prepared.fitting.features[names])
    transform = lambda view: pd.DataFrame(scaler.transform(view.features[names]),
                                          index=view.features.index,columns=names)
    estimator = XGBClassifier(**PARAMETERS)
    estimator.fit(transform(prepared.fitting),prepared.fitting.labels,
                  eval_set=[(transform(prepared.validation),prepared.validation.labels)],verbose=False)
    return selector,scaler,estimator,tuple(names),transform


def local_tracking(root, publication, protocol, receipt):
    """Create a fresh local SQLite destination; no global tracking URI is changed."""
    from mlflow.tracking import MlflowClient
    from src.tracking.research_record import build_research_tracking_record
    from src.tracking.research_mlflow import TrackingDestination, BoundMlflowClient, link_research_record
    from src.tracking.research_mlflow_client import ResearchMlflowClient
    from src.tracking.research_reconciliation import reconcile_research_tracking
    store = root/"tracking"; store.mkdir(); artifacts=store/"artifacts";artifacts.mkdir()
    os.environ["_MLFLOW_SERVER_ARTIFACT_ROOT"] = artifacts.as_uri()
    uri = "sqlite:///"+str(store/"backend.sqlite")
    real = MlflowClient(tracking_uri=uri,registry_uri=uri,workspace_store_uri=uri)
    experiment = real.create_experiment("synthetic-public-v1",artifact_location=artifacts.as_uri())
    destination = TrackingDestination("synthetic-local",experiment)
    bound = BoundMlflowClient(destination,ResearchMlflowClient(real))
    record = build_research_tracking_record(publication,publication.published_path,
        runtime_metadata=dict(ticker="SYNTH",asset_id=protocol.asset_id,fitting_attempt_id=receipt.attempt_id,
            pipeline_run_id="synthetic-run",training_status="success",reporting_status="success",
            archive_alias="synthetic-archive",tracking_destination_id=destination.destination_id))
    journal=store/"journal";journal.mkdir()
    outcome=link_research_record(record,client=bound,destination=destination,journal_root=journal,enabled=True)
    if outcome.status != "linked":
        raise RuntimeError("Local linkage failed: "+outcome.status)
    report=reconcile_research_tracking(archive_root=publication.archive_root,archive_alias="synthetic-archive",
        journal_root=journal,destination=destination,client=bound)
    result=dict(linkage_status=outcome.status,reconciliation=report.to_dict())
    real._tracking_client.store.engine.dispose()
    return result


def json_safe(value):
    """Use JSON null for undefined metrics, rather than non-standard NaN literals."""
    if isinstance(value,dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value,(datetime,pd.Timestamp)): return value.isoformat()
    if isinstance(value,np.generic): return json_safe(value.item())
    if isinstance(value,float) and not math.isfinite(value): return None
    return value


def run_experiment(output, *, tracking=False):
    """Own a new output directory; never overwrite an existing experiment."""
    output=Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Output must be a new directory; existing paths are never reused.")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.mkdir(mode=0o700)
    output=output.resolve(strict=True)
    with offline_guard(output):
        frame,prices=generate_ohlcv()
        protocol=build_protocol(output,frame,prices)
        prepared=prepare_research_inputs(protocol,output)
        selector,scaler,estimator,names,transform=fit_states(prepared)
        fit_completed=datetime.now(UTC)
        state=inspect_research_fit(protocol,prepared,experiment_id="public-synthetic-v1",selector=selector,
            scaler=scaler,estimator=estimator,selected_features=names)
        held=set(protocol.partitions.held_out)
        rows=tuple(SourceRow(row.observation_at,row.observation_at,row.observation_at+timedelta(minutes=1),
            row.target_end_timestamp,None if row.row_id in held else row.target_end_timestamp+timedelta(minutes=2))
            for row in prepared.row_bindings)
        fitting=prepared.fitting.features.index; validation=prepared.validation.features.index
        selection=max(row.labels_available_at for row in rows if row.row_id in fitting)+timedelta(minutes=1)
        early=max(row.labels_available_at for row in rows if row.row_id in validation)+timedelta(minutes=1)
        split=SimpleNamespace(source_index=pd.DatetimeIndex([row.observation_at for row in prepared.row_bindings
            if row.row_id in set(protocol.partitions.fitting+protocol.partitions.inner_purge+protocol.partitions.validation)]),
            fitting_index=fitting,validation_index=validation,purged_count=5)
        manifest=FrozenModelManifest(protocol.asset_id,state.binding.model_id,"public-synthetic-v1",rows,
            tuple(fitting),tuple(prepared.inner_purged.features.index),tuple(prepared.outer_purged.features.index),
            tuple(validation),tuple(prepared.held_out.features.index),rows[0].observation_at,validation[0],
            prepared.held_out.features.index[0],"5d_class",5,.005,names,selection,
            (FittedStateAvailability("selector",selection),FittedStateAvailability("scaler",selection)),
            True,early,timedelta(minutes=10),timedelta(minutes=1),fit_completed)
        test=transform(prepared.held_out)
        predicted=estimator.predict(test); probabilities=estimator.predict_proba(test)
        capture=capture_xgboost_predictions(test_index=test.index,feature_names=names,model_feature_names=names,
            predictions=pd.Series(predicted,index=test.index),probabilities=pd.DataFrame(probabilities,index=test.index,
            columns=estimator.classes_),estimator=estimator,ticker="SYNTH",asset_id=protocol.asset_id,
            temporal_split=split,actual_prediction_generated_at=datetime.now(UTC))
        asset=AssetSpec(protocol.asset_id,"SYNTH","equity","synthetic-weekday-utc-v1","USD")
        fresh=bind_fresh_capture(capture,run_id="synthetic-attempt",model=state.binding,asset=asset)
        attempt=ResearchFitAttempt("synthetic-attempt","SYNTH",protocol.asset_id,"public-synthetic-v1",
            protocol.protocol_id,prepared.training_input_id,manifest.split_id,split,fit_completed)
        receipt=create_research_fit_receipt(protocol,prepared,attempt=attempt,selector=selector,scaler=scaler,
            estimator=estimator,selected_features=names,temporal_manifest=manifest,capture=capture,fresh_capture_receipt=fresh)
        owned=capture_research_bundle(protocol,prepared,receipt,ticker="SYNTH",run_id="synthetic-run",research_root=output,
            selector=selector,scaler=scaler,estimator=estimator,capture=capture,fresh_capture_receipt=fresh)
        candidate=finalize_research_bundle(owned,ticker="SYNTH",run_id="synthetic-run",attempt_id=attempt.attempt_id,
            receipt_id=receipt.receipt_id,run_status="success")
        archive=output/"archive";archive.mkdir()
        publication=publish_prepared_research_bundle(candidate,ResearchPublicationContext(archive,RESEARCH_ARCHIVE_POLICY),
            ticker="SYNTH",run_id="synthetic-run",attempt_id=attempt.attempt_id,receipt_id=receipt.receipt_id,run_status="success")
        replay=replay_research_bundle(publication.published_path,publication.bundle_id,enabled=True)
        if replay.status != "verified":
            raise RuntimeError("Functional replay rejected: "+replay.status+" "+repr(replay.checks))
        # No target or target-return column is accepted by this signal boundary.
        by_close={row.close_at:row for row in prices.itertuples()}
        signals=[]
        for row in capture.records:
            market=by_close[row.feature_row_id]
            decision=row.feature_row_id+timedelta(minutes=2)
            manifest.validate_decision(decision)
            if not manifest.model_information_cutoff < decision:
                raise ValueError("Information cutoff must strictly precede decisions.")
            signals.append(Signal(market.session_id,decision,"long" if row.predicted_class==2 else "flat"))
        bars=[MarketBar("SYNTH",row.session_id,row.open_at,row.close_at,float(row.open),float(row.close),
            high=float(max(row.open,row.close)*1.003),low=float(min(row.open,row.close)*.997),volume=1000.)
            for row in prices.iloc[120:180].itertuples()]
        comparisons={}
        for policy in ("fixed_horizon","replace_next_open"):
            config=ExecutionConfig("SYNTH",initial_capital=10000,allocation_fraction=.5,holding_sessions=5,
                fee_rate=.001,slippage_rate=.0005,position_policy=policy,
                terminal_policy="liquidate_at_final_close",terminal_session_id="session-179")
            executed=ExecutionEngineV1(config).run(bars,signals)
            performance=calculate_performance(executed,PerformanceConfig(252,.02))
            if not performance.completeness.full_window_comparison_eligible:
                raise RuntimeError("Incomplete execution windows cannot be compared.")
            np.testing.assert_allclose(executed.ledger.equity,
                executed.ledger.cash+executed.ledger.position_quantity*executed.ledger.close_price.fillna(0))
            executed.ledger.to_csv(output/(policy+"-ledger.csv"),index=False)
            executed.fills.to_csv(output/(policy+"-fills.csv"),index=False)
            comparisons[policy]=dict(portfolio=performance.portfolio_metrics,trades=performance.trade_metrics,
                turnover=performance.turnover_metrics,complete=performance.completeness.full_window_comparison_eligible)
        # Held-out labels are consumed only here, after capture and signal creation.
        evaluation=dict(accuracy=float(accuracy_score(prepared.held_out.labels,predicted)),
            log_loss=float(log_loss(prepared.held_out.labels,probabilities,labels=estimator.classes_)),
            class_labels=[0,1,2],confusion_matrix=confusion_matrix(prepared.held_out.labels,predicted,labels=[0,1,2]).tolist())
        summary=dict(scope="synthetic_only_not_investment_results",profile="apf-public-v1",
            protocol_id=protocol.protocol_id,training_input_id=prepared.training_input_id,receipt_id=receipt.receipt_id,
            bundle_id=publication.bundle_id,selected_features=names,
            counts=dict(fitting=len(fitting),inner_purged=5,validation=len(validation),outer_purged=5,held_out=len(test)),
            model_information_cutoff=manifest.model_information_cutoff,
            earliest_historical_decision=manifest.earliest_historical_decision,actual_fit_completed_at=fit_completed,
            evaluation=evaluation,replay=dict(status=replay.status,rows=replay.compared_rows,
                maximum_probability_difference=replay.maximum_absolute_error,class_mismatches=replay.class_mismatched_rows),
            execution=comparisons,execution_certified=False,historical_pit_verified=False,
            provider_evidence="none; synthetic fixtures only",tracking_enabled=tracking,
            tracking=local_tracking(output,publication,protocol,receipt) if tracking else dict(status="disabled"))
        summary=json_safe(summary)
        (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
        return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path,help="New explicit output directory, never an existing archive")
    parser.add_argument("--track-local",action="store_true",help="Create a new isolated local SQLite MLflow backend")
    args=parser.parse_args()
    print(json.dumps(run_experiment(args.output,tracking=args.track_local),indent=2))


if __name__ == "__main__":
    main()
