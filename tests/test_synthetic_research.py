"""Independent public-only causal and installed-demo regression contracts."""
from dataclasses import replace
from datetime import timedelta
import importlib.util
import os
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import pytest

from examples.synthetic_research import (
    build_protocol, fit_states, generate_ohlcv, offline_guard, run_experiment,
)
from src.research.input_adapter import prepare_research_inputs
from src.evaluation.execution.engine import ExecutionEngineV1
from src.evaluation.execution.contracts import ExecutionConfig, MarketBar, Signal


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    # The demonstration intentionally sets an explicit local artifact root.
    # Its test fixture owns and restores that process environment on all exits.
    key = "_MLFLOW_SERVER_ARTIFACT_ROOT"
    previous = os.environ.get(key)
    try:
        yield run_experiment(tmp_path_factory.mktemp("portfolio-demo-parent")/"fresh",tracking=True)
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def test_real_training_receipt_bundle_replay_and_local_tracking(completed):
    assert completed["counts"] == dict(fitting=60,inner_purged=5,validation=19,outer_purged=5,held_out=55)
    assert completed["receipt_id"].startswith("apf-research-fit-receipt-v1-")
    assert completed["bundle_id"].startswith("apf-research-bundle-v1-")
    assert completed["replay"] == dict(status="verified",rows=55,maximum_probability_difference=0.,class_mismatches=0)
    assert completed["tracking"]["linkage_status"] == "linked"
    assert completed["evaluation"]["class_labels"] == [0,1,2]
    assert sum(map(sum,completed["evaluation"]["confusion_matrix"])) == 55
    assert not completed["execution_certified"] and not completed["historical_pit_verified"]
    assert completed["actual_fit_completed_at"] != completed["model_information_cutoff"]


def test_identical_complete_windows_for_both_execution_policies(completed):
    assert set(completed["execution"]) == {"fixed_horizon","replace_next_open"}
    assert all(value["complete"] for value in completed["execution"].values())


def test_generator_prefix_invariance_and_separate_execution_feed():
    before,market=generate_ohlcv(180)
    after,future=generate_ohlcv(200)
    pd.testing.assert_frame_equal(before,after.iloc[:180])
    pd.testing.assert_frame_equal(market,future.iloc[:180])
    assert not np.array_equal(before.close.to_numpy(),market.close.to_numpy())


def test_explicit_purge_counts_and_original_maturity(tmp_path):
    frame,prices=generate_ohlcv()
    protocol=build_protocol(tmp_path,frame,prices)
    prepared=prepare_research_inputs(protocol,tmp_path)
    first_val=prepared.validation.row_bindings[0].observation_at
    first_test=prepared.held_out.row_bindings[0].observation_at
    assert all(row.target_end_timestamp<first_val for row in prepared.fitting.row_bindings)
    assert all(row.target_end_timestamp<first_test for row in prepared.validation.row_bindings)
    assert prepared.inner_purged.row_bindings[0].target_end_timestamp==first_val
    assert prepared.outer_purged.row_bindings[0].target_end_timestamp==first_test
    assert len(prepared.inner_purged.row_ids)==len(prepared.outer_purged.row_ids)==5
    assert prepared.fitting.row_bindings[0].target_end_source_id=="r36"


def test_future_prices_cannot_change_training_features_or_source_context(tmp_path):
    frame,prices=generate_ohlcv()
    left=tmp_path/"left";right=tmp_path/"right";left.mkdir();right.mkdir()
    p=build_protocol(left,frame,prices)
    original=prepare_research_inputs(p,left)
    changed=frame.copy();changed.loc[160:,"close"]*=1.1
    other=prepare_research_inputs(build_protocol(right,changed,prices),right)
    pd.testing.assert_frame_equal(original.fitting.features,other.fitting.features)
    assert original.source_training_input_id==other.source_training_input_id
    assert original.held_out.row_ids==other.held_out.row_ids


def test_selection_scaling_and_fit_receive_only_authorized_rows(tmp_path,monkeypatch):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import RobustScaler
    from xgboost import XGBClassifier
    frame,prices=generate_ohlcv()
    prepared=prepare_research_inputs(build_protocol(tmp_path,frame,prices),tmp_path)
    permitted=prepared.fitting.features.index
    seen=[]
    for cls in (RandomForestClassifier,RobustScaler,XGBClassifier):
        original=cls.fit
        def inspect(self,X,*args,_original=original,**kwargs):
            assert X.index.equals(permitted)
            assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes)
            assert not any(name.startswith("target") for name in X.columns)
            if isinstance(self,XGBClassifier):
                assert kwargs["eval_set"][0][0].index.equals(prepared.validation.features.index)
            seen.append(type(self).__name__)
            return _original(self,X,*args,**kwargs)
        monkeypatch.setattr(cls,"fit",inspect)
    fit_states(prepared)
    assert seen==["RandomForestClassifier","RobustScaler","XGBClassifier"]


def test_validation_and_test_label_changes_cannot_influence_selection(tmp_path):
    frame,prices=generate_ohlcv()
    prepared=prepare_research_inputs(build_protocol(tmp_path,frame,prices),tmp_path)
    _,scaler,_,names,_=fit_states(prepared)
    # Change the actual validation/held-out partitions supplied to fit_states.
    # Validation may change early stopping, but not fitted selection/scaling.
    changed=prepared.validation.labels.copy();changed[:]=2
    held=prepared.held_out.labels.copy();held[:]=0
    altered=replace(prepared,validation=replace(prepared.validation,_labels=changed),
                    held_out=replace(prepared.held_out,_labels=held))
    _,other_scaler,_,other_names,_=fit_states(altered)
    assert other_names==names
    np.testing.assert_array_equal(scaler.center_,other_scaler.center_)
    np.testing.assert_array_equal(scaler.scale_,other_scaler.scale_)
    assert not altered.validation.labels.equals(prepared.validation.labels)
    assert not altered.held_out.labels.equals(prepared.held_out.labels)


def test_existing_output_is_rejected_without_mutation(tmp_path):
    target=tmp_path/"existing";target.mkdir();(target/"sentinel").write_text("preserve")
    with pytest.raises(ValueError,match="new directory"):
        run_experiment(target)
    assert (target/"sentinel").read_text()=="preserve"


def test_network_guard_fails_closed(tmp_path):
    with offline_guard(tmp_path.resolve()):
        with pytest.raises(RuntimeError,match="Network access"):
            socket.getaddrinfo("example.invalid",443)


def test_no_private_pipeline_or_provider_is_packaged():
    for name in ("src.pipeline","src.data.yfinance_acquisition","src.config"):
        assert importlib.util.find_spec(name) is None


def test_next_open_fills_and_cash_reconcile_independently():
    from datetime import datetime,timezone
    days=pd.bdate_range("2024-01-01",periods=3,tz="UTC")
    bars=[MarketBar("SYNTH",f"s{i}",day.to_pydatetime()+timedelta(hours=9),
        day.to_pydatetime()+timedelta(hours=16),price,price) for i,(day,price) in enumerate(zip(days,[100,100,110]))]
    result=ExecutionEngineV1(ExecutionConfig("SYNTH",initial_capital=1000,holding_sessions=2,
        fee_rate=.01,terminal_policy="liquidate_at_final_close",terminal_session_id="s2")).run(
        bars,[Signal("s0",bars[0].close_timestamp+timedelta(minutes=1),"long")])
    assert result.fills.iloc[0].timestamp==bars[1].open_timestamp
    assert result.fills.iloc[0].quantity==9
    assert result.ledger.iloc[-1].equity==pytest.approx(1071.10)
    assert result.ledger.iloc[-1].cash==pytest.approx(1071.10)
