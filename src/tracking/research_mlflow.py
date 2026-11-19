"""Explicit secondary indexing through an injected MLflow-compatible client.

No MLflow import, fluent contexts, experiment creation, model operations or
archive access occur. create_run receives the entire frozen tag payload. Runs
already present are read-only: incomplete/conflicting tags are never repaired
with unconditional set_tag calls (MLflow has no atomic absent-only tag write).

A caller binds the client's destination explicitly; this is a trusted runtime
declaration, not server authentication. Search/get/create must provide the
standard MLflow Run.info/Run.data.tags and paged-list token interfaces. Bounded
search includes deleted runs. The remote backend may be eventually consistent:
locks protect one local journal, not all independent clients or remote writers.
No exactly-once creation claim is made. An ambiguous create is never retried.
"""
from dataclasses import dataclass, field, replace
from typing import Any

from .research_record import ResearchTrackingRecordV1, ResearchTrackingError, _alias
from .research_linkage_journal import LinkageJournal, LinkageIntent, JournalError, new_intent, opaque_id


@dataclass(frozen=True)
class TrackingDestination:
    destination_id: str
    experiment_id: str

    def __post_init__(self):
        if not _alias(self.destination_id) or not opaque_id(self.experiment_id):
            raise ValueError("invalid_tracking_destination")


@dataclass(frozen=True)
class BoundMlflowClient:
    """Bind an already configured client; construction performs no I/O.

    Only search_runs, create_run and get_run are used. Metadata logging is part
    of create_run(tags=...). MLflow's technical run termination is not changed;
    linked describes verified index metadata, never training/server run status.
    """
    destination: TrackingDestination
    client: Any = field(repr=False)


@dataclass(frozen=True)
class LinkageOutcome:
    status: str
    intent: LinkageIntent | None = None
    diagnostics: tuple[str, ...] = ()


class _RemoteProtocolError(ValueError):
    pass


def _require(value):
    if not value:
        raise _RemoteProtocolError("invalid_remote_response")


def _remote_run(run, experiment_id):
    info = getattr(run, "info", None)
    run_id = getattr(info, "run_id", None)
    _require(opaque_id(run_id) and getattr(info, "experiment_id", None) == experiment_id)
    _require(getattr(info, "lifecycle_stage", None) == "active")
    tags = getattr(getattr(run, "data", None), "tags", None)
    _require(type(tags) is dict and len(tags) <= 512)
    # Unknown non-APF tags are neither trusted nor copied to the journal.
    apf = {key: value for key, value in tags.items() if type(key) is str and key.startswith("apf.")}
    _require(all(type(value) is str and len(key) <= 192 and len(value) <= 4096 for key, value in apf.items()))
    return run_id, apf


def _search(client, destination, key):
    ids, tokens, token = set(), set(), None
    for _ in range(100):
        page = client.search_runs(experiment_ids=[destination.experiment_id],
            filter_string=f"tags.`apf.linkage_key` = '{key}'", run_view_type=3,  # MLflow ViewType.ALL.
            max_results=100, page_token=token)
        _require(isinstance(page, (list, tuple)) and len(page) <= 100)
        for run in page:
            info = getattr(run, "info", None)
            identity = getattr(info, "run_id", None)
            _require(opaque_id(identity) and getattr(info, "experiment_id", None) == destination.experiment_id)
            ids.add(identity)
        _require(len(ids) <= 1000)
        token = getattr(page, "token", None)
        if token is None:
            return ids
        _require(type(token) is str and 0 < len(token) <= 4096 and token not in tokens)
        tokens.add(token)
    raise _RemoteProtocolError("search_page_limit")


def link_research_record(record=None, *, client=None, destination=None, journal_root=None, enabled=False):
    """Persist intent before client I/O; return linkage outcome independently.

    enabled must be exactly True. A fresh intent permits at most one create.
    Creating/crash/timeout history keeps create_may_exist permanently true.
    Retrying is reconciliation only, even after a zero-result search. Operator
    intervention is needed for unresolved uncertainty or partial metadata.
    No publication/training/execution callback is accepted by this interface.
    """
    if enabled is False:
        return LinkageOutcome("disabled")
    if enabled is not True:
        return LinkageOutcome("rejected", diagnostics=("explicit_activation_required",))
    if type(record) is not ResearchTrackingRecordV1:
        return LinkageOutcome("rejected", diagnostics=("invalid_research_record",))
    try:
        record = ResearchTrackingRecordV1.from_json(record.canonical_json())
    except ResearchTrackingError:
        return LinkageOutcome("rejected", diagnostics=("invalid_research_record",))
    if (type(destination) is not TrackingDestination or type(client) is not BoundMlflowClient
            or client.destination != destination or destination.destination_id != record.tracking_destination_id):
        return LinkageOutcome("rejected", diagnostics=("tracking_destination_or_experiment_mismatch",))
    intended = new_intent(record, destination.experiment_id)
    current = None
    try:
        with LinkageJournal(journal_root).transaction(record.linkage_key) as transaction:
            current = transaction.read()
            if current is None:
                current = transaction.write(intended, expected_revision=0)
            elif (current.record_id, current.record_json, current.metadata_payload, current.destination_id, current.experiment_id) != (
                    intended.record_id, intended.record_json, intended.metadata_payload, intended.destination_id, intended.experiment_id):
                return LinkageOutcome("conflict", current, ("journal_intent_conflict",))

            if any(not callable(getattr(client.client, method, None)) for method in ("search_runs", "create_run", "get_run")):
                return LinkageOutcome("rejected", current, ("invalid_client_boundary",))

            def advance(status, phase, diagnostics=(), **changes):
                nonlocal current
                updated = replace(current, status=status, phase=phase, diagnostics=diagnostics,
                                  revision=current.revision + 1, **changes)
                current = transaction.write(updated, expected_revision=current.revision)
                return LinkageOutcome(status, current, diagnostics)

            def failure(code, phase, error):
                diagnostic = (code, "remote_timeout") if isinstance(error, TimeoutError) else (code,)
                return advance("linkage_uncertain" if current.create_may_exist else "failed", phase, diagnostic)

            advance("searching", "search")
            try:
                ids = _search(client.client, destination, record.linkage_key)
            except Exception as error:
                return failure("remote_search_failed", "search", error)
            if current.mlflow_run_id is not None:
                ids.add(current.mlflow_run_id)
            if not ids:
                if current.create_may_exist:
                    return advance("linkage_uncertain", "search", ("create_visibility_uncertain",))
                advance("creating", "create", create_may_exist=True)
                try:
                    created = client.client.create_run(experiment_id=destination.experiment_id,
                        tags=dict(current.metadata_payload), run_name="APF research archive reference")
                    info = getattr(created, "info", None)
                    created_id = getattr(info, "run_id", None)
                    _require(opaque_id(created_id) and getattr(info, "experiment_id", None) == destination.experiment_id)
                except Exception as error:
                    return failure("remote_create_ambiguous", "create", error)
                # A successful response supplies one candidate ID, not proof of
                # metadata completeness. Search again for visible competitors.
                advance("searching", "search")
                try:
                    ids = _search(client.client, destination, record.linkage_key)
                except Exception as error:
                    return failure("remote_search_failed", "search", error)
                ids.add(created_id)
            advance("searching", "readback", create_may_exist=True)
            desired = dict(current.metadata_payload)
            complete, partial, conflicting = [], [], []
            for run_id in sorted(ids):
                try:
                    run = client.client.get_run(run_id)
                    identity, tags = _remote_run(run, destination.experiment_id)
                    _require(identity == run_id)
                except Exception as error:
                    return failure("remote_readback_failed", "readback", error)
                if any(key not in desired or value != desired[key] for key, value in tags.items()):
                    conflicting.append(run_id)
                elif tags != desired:
                    partial.append(run_id)
                else:
                    complete.append(run_id)
            if conflicting:
                return advance("conflict", "readback", ("remote_metadata_conflict",))
            if len(ids) > 1:
                return advance("duplicate", "readback", ("duplicate_remote_runs",))
            if partial:
                return advance("failed", "readback", ("remote_metadata_incomplete",))
            if len(complete) != 1:
                return advance("linkage_uncertain", "readback", ("create_visibility_uncertain",))
            return advance("linked", "complete", mlflow_run_id=complete[0], create_may_exist=True)
    except JournalError as error:
        status = "journal_busy" if error.code == "journal_busy" else "journal_uncertain" if error.uncertain else "journal_failed"
        return LinkageOutcome(status, current, (error.code,))
