"""Declared synthetic records, fake clients and temporary journals only.

No actual MLflow library/client, model, archive publisher or database is used.
The fixture is not authenticated publication or historical market evidence.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from src.tracking import research_mlflow as api
from src.tracking import research_linkage_journal as journal
from src.tracking.research_record import ResearchTrackingRecordV1, PREFIXES, CHECKS, MODEL_PROFILE
from src.research.bundle_publication import RESEARCH_ARCHIVE_POLICY, ACCEPTED


@pytest.fixture
def record():
    identities={name:prefix+hashlib.sha256(name.encode()).hexdigest() for name,prefix in PREFIXES.items()}
    return ResearchTrackingRecordV1(**identities,asset_id="research:SYNTH",fitting_attempt_id="fit-001",
        research_experiment_id="synthetic-research",ticker="SYNTH",pipeline_run_id="pipeline-001",
        archive_alias="synthetic_archive",tracking_destination_id="research_index",model_type="xgboost",
        estimator_class="xgboost.sklearn.XGBClassifier",estimator_version="3.2.0",model_profile=MODEL_PROFILE,
        training_status="success",reporting_status="success",publisher_outcome="published",
        archive_acceptance_policy=RESEARCH_ARCHIVE_POLICY,archive_status=ACCEPTED,research_verification_status="incomplete",
        decision_time_evidence="incomplete",historical_pit_status="unknown",execution_eligibility_status="unassessed",
        functional_replay_status="not_assessed",provenance_origin="synthetic_fixture",verification_summary=CHECKS)


class Page(list):
    def __init__(self,values,token=None):
        super().__init__(values)
        self.token=token


class FakeClient:
    def __init__(self,store=None):
        self.store=store if store is not None else {}
        self.calls=[]
        self.failure=None
        self.page_size=100
        self.observe=None
        self.lock=threading.RLock()
        self.hidden=False

    def observe_call(self,name):
        self.calls.append(name)
        if self.observe:self.observe(name)

    def run(self,identity):
        item=self.store[identity]
        return SimpleNamespace(info=SimpleNamespace(run_id=identity,experiment_id=item['experiment'],lifecycle_stage=item.get('lifecycle','active')),
                               data=SimpleNamespace(tags=dict(item['tags'])))

    def search_runs(self,*,experiment_ids,filter_string,run_view_type,max_results,page_token):
        self.observe_call('search')
        assert experiment_ids==['experiment-1'] and run_view_type==3 and max_results==100
        if self.failure=='search':raise ConnectionError('private-token-value-never-persist')
        if self.failure=='timeout_search':raise TimeoutError('private-token-value-never-persist')
        key=filter_string.split("'")[1]
        with self.lock:
            ids=[] if self.hidden else [identity for identity,item in self.store.items()
                if item['experiment']==experiment_ids[0] and item['tags'].get('apf.linkage_key')==key]
            start=int(page_token or 0)
            chosen=ids[start:start+self.page_size]
            token=str(start+self.page_size) if start+self.page_size<len(ids) else None
            return Page([self.run(identity) for identity in chosen],token)

    def create_run(self,*,experiment_id,tags,run_name):
        self.observe_call('create')
        assert experiment_id=='experiment-1' and run_name=='APF research archive reference'
        if self.failure=='create':raise ConnectionError('private-token-value-never-persist')
        with self.lock:
            identity='run-'+str(len(self.store)+1)
            self.store[identity]={'experiment':experiment_id,'tags':dict(tags)}
            if self.failure in {'partial','tag_failure'}:
                self.store[identity]['tags'].pop('apf.verification.temporal_provenance')
            if self.failure=='tag_failure':raise RuntimeError('private-token-value-never-persist')
            if self.failure=='lost_response':raise TimeoutError('private-token-value-never-persist')
            return self.run(identity)

    def get_run(self,identity):
        self.observe_call('read')
        if self.failure=='read':raise TimeoutError('private-token-value-never-persist')
        return self.run(identity)

    def set_tag(self,*args,**kwargs):
        pytest.fail('Existing remote research metadata must never be overwritten')


@pytest.fixture
def case(record,tmp_path):
    root=(tmp_path/'journal').resolve()
    root.mkdir()
    destination=api.TrackingDestination('research_index','experiment-1')
    fake=FakeClient()
    return SimpleNamespace(record=record,root=root,destination=destination,fake=fake,
        client=api.BoundMlflowClient(destination,fake))


def link(c,**changes):
    args=dict(record=c.record,client=c.client,destination=c.destination,journal_root=c.root,enabled=True)
    args.update(changes)
    return api.link_research_record(**args)


def saved(c):
    return journal.LinkageIntent.from_json((c.root/(c.record.linkage_key+'.json')).read_bytes())


def seed(c,identity='existing',**changes):
    tags=dict(journal.metadata_payload(c.record,c.destination.experiment_id))
    tags.update(changes)
    c.fake.store[identity]={'experiment':c.destination.experiment_id,'tags':tags}


def test_disabled_path_does_not_validate_record_or_touch_client_or_files(case):
    assert api.link_research_record(object(),client=object(),journal_root=object()).status=='disabled'
    assert not list(case.root.iterdir())
    assert not case.fake.calls
    assert link(case,enabled=1).status=='rejected'


def test_success_has_durable_pending_intent_before_every_client_call(case):
    c=case
    before=c.record.canonical_json()
    def observed(name):
        entry=saved(c)
        assert entry.record_id==c.record.record_id and entry.record_json==before
        assert entry.record_sha256==hashlib.sha256(before.encode()).hexdigest()
        if name=='create':assert entry.status=='creating' and entry.phase=='create' and entry.create_may_exist
        else:assert entry.status=='searching' and entry.phase in {'search','readback'}
    c.fake.observe=observed
    result=link(c)
    assert result.status=='linked' and result.intent.mlflow_run_id=='run-1'
    assert result.intent==saved(c) and result.intent.create_may_exist
    assert c.fake.calls==['search','create','search','read']
    tags=c.fake.store['run-1']['tags']
    for key,value in {'apf.decision_time_evidence':'incomplete','apf.historical_pit_status':'unknown',
        'apf.execution_eligibility_status':'unassessed','apf.functional_replay_status':'not_assessed'}.items():assert tags[key]==value
    assert c.record.canonical_json()==before


def test_exact_existing_run_reused_and_repeat_linkage_is_idempotent(case):
    seed(case)
    first=link(case)
    second=link(case)
    assert first.status==second.status=='linked' and first.intent.mlflow_run_id==second.intent.mlflow_run_id=='existing'
    assert 'create' not in case.fake.calls and len(case.fake.store)==1
    assert second.intent.revision>first.intent.revision


@pytest.mark.parametrize('change',['destination','experiment','record','root','activation'])
def test_explicit_binding_and_input_validation_precede_client_io(case,change):
    options={'destination':{'destination':api.TrackingDestination('another_index','experiment-1')},
        'experiment':{'destination':api.TrackingDestination('research_index','experiment-2')},
        'record':{'record':None},'root':{'journal_root':None},'activation':{'enabled':'true'}}[change]
    result=link(case,**options)
    assert result.status in {'rejected','journal_failed'} and not case.fake.calls
    assert not list(case.root.iterdir())


def test_missing_archive_reference_in_tampered_record_is_rejected(case):
    object.__setattr__(case.record,'archive_alias','')
    assert link(case).status=='rejected' and not case.fake.calls


@pytest.mark.parametrize('field',['apf.record_id','apf.record_sha256','apf.protocol_id','apf.asset_id','apf.archive_alias',
    'apf.historical_pit_status','apf.tracking_experiment_id','apf.unknown'])
def test_conflicting_remote_payload_is_never_overwritten(case,field):
    seed(case,**{field:'private-token-value-never-persist'})
    result=link(case)
    assert result.status=='conflict' and result.intent.mlflow_run_id is None
    assert 'create' not in case.fake.calls
    assert case.fake.store['existing']['tags'][field]=='private-token-value-never-persist'
    assert 'private-token-value-never-persist' not in saved(case).canonical_json()


def test_all_search_pages_are_checked_before_duplicate_decision(case):
    seed(case,'one');seed(case,'two');seed(case,'three')
    case.fake.page_size=1
    result=link(case)
    assert result.status=='duplicate' and result.intent.mlflow_run_id is None
    assert case.fake.calls==['search','search','search','read','read','read']


@pytest.mark.parametrize('failure',['search','timeout_search','create','lost_response','read','partial','tag_failure'])
def test_client_failures_preserve_explicit_state_without_sensitive_messages(case,failure):
    case.fake.failure=failure
    result=link(case)
    expected='failed' if failure in {'search','timeout_search','partial'} else 'linkage_uncertain'
    assert result.status==expected
    entry=saved(case)
    assert 'private-token-value-never-persist' not in entry.canonical_json()
    assert entry.record_id==case.record.record_id and entry.bundle_id==case.record.bundle_id
    assert entry.mlflow_run_id is None
    if failure not in {'search','timeout_search'}:assert entry.create_may_exist


def test_lost_create_response_recovers_one_existing_run_without_new_create(case):
    case.fake.failure='lost_response'
    assert link(case).status=='linkage_uncertain' and len(case.fake.store)==1
    case.fake.failure=None
    assert link(case).status=='linked'
    assert case.fake.calls.count('create')==1


def test_zero_results_after_ambiguous_creation_never_authorize_another_create(case):
    case.fake.failure='create'
    assert link(case).status=='linkage_uncertain'
    case.fake.failure=None
    for _ in range(3):assert link(case).status=='linkage_uncertain'
    assert case.fake.calls.count('create')==1 and not case.fake.store


def test_delayed_visibility_remains_uncertain_until_later_reconciliation(case):
    case.fake.failure='lost_response'
    assert link(case).status=='linkage_uncertain'
    case.fake.failure=None
    case.fake.hidden=True
    assert link(case).status=='linkage_uncertain'
    case.fake.hidden=False
    assert link(case).status=='linked' and case.fake.calls.count('create')==1


def test_partial_metadata_is_not_automatically_repaired_or_overwritten(case):
    case.fake.failure='partial'
    assert link(case).status=='failed'
    case.fake.failure=None
    assert link(case).status=='failed' and case.fake.calls.count('create')==1
    # Explicit synthetic operator repair, outside the linkage service.
    case.fake.store['run-1']['tags'].update(dict(saved(case).metadata_payload))
    assert link(case).status=='linked' and case.fake.calls.count('create')==1


def test_crash_like_creating_state_is_reconciliation_only(case):
    entry=journal.new_intent(case.record,case.destination.experiment_id)
    with journal.LinkageJournal(case.root).transaction(case.record.linkage_key) as tx:
        tx.write(entry,expected_revision=0)
        tx.write(replace(entry,status='creating',phase='create',create_may_exist=True,revision=2),expected_revision=1)
    assert link(case).status=='linkage_uncertain' and 'create' not in case.fake.calls


def test_journal_failure_before_any_client_io(case,monkeypatch):
    def fail(*args,**kwargs):raise journal.JournalError('journal_write_failed')
    monkeypatch.setattr(journal._Transaction,'write',fail)
    result=link(case)
    assert result.status=='journal_failed' and not case.fake.calls


def test_journal_failure_after_remote_create_recovers_without_recreating(case,monkeypatch):
    original=journal._Transaction.write
    def fail(self,entry,**kwargs):
        if case.fake.calls.count('create') and entry.phase=='search':raise journal.JournalError('journal_write_failed')
        return original(self,entry,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(journal._Transaction,'write',fail)
        assert link(case).status=='journal_failed'
    assert saved(case).status=='creating' and saved(case).create_may_exist
    assert link(case).status=='linked' and case.fake.calls.count('create')==1


@pytest.mark.parametrize('fault',['file_sync','directory_sync','rename_after_commit'])
def test_atomic_write_failures_prevent_client_io_and_report_uncertainty(case,monkeypatch,fault):
    import stat
    fsync,rename=journal.os.fsync,journal.os.replace
    def failed_sync(fd):
        is_dir=stat.S_ISDIR(journal.os.fstat(fd).st_mode)
        if (fault=='directory_sync')==is_dir:raise OSError('private-token-value-never-persist')
        return fsync(fd)
    def renamed(*args,**kwargs):
        rename(*args,**kwargs)
        raise OSError('private-token-value-never-persist')
    if fault=='rename_after_commit':monkeypatch.setattr(journal.os,'replace',renamed)
    else:monkeypatch.setattr(journal.os,'fsync',failed_sync)
    result=link(case)
    assert result.status==('journal_failed' if fault=='file_sync' else 'journal_uncertain')
    assert not case.fake.calls
    if fault!='file_sync':assert saved(case).status=='pending'


def test_changed_archive_alias_with_same_linkage_key_is_an_intent_conflict(case):
    assert link(case).status=='linked'
    before=saved(case).canonical_json()
    calls=len(case.fake.calls)
    changed=replace(case.record,archive_alias='another_archive')
    assert changed.linkage_key==case.record.linkage_key and changed.record_id!=case.record.record_id
    assert link(case,record=changed).status=='conflict'
    assert saved(case).canonical_json()==before and len(case.fake.calls)==calls


def test_existing_journal_cannot_switch_experiments(case):
    assert link(case).status=='linked'
    destination=api.TrackingDestination('research_index','experiment-2')
    assert link(case,destination=destination,client=api.BoundMlflowClient(destination,case.fake)).status=='conflict'
    assert saved(case).experiment_id=='experiment-1'


def test_concurrent_local_calls_and_separate_journal_instances_do_not_lose_updates(case):
    entered,release=threading.Event(),threading.Event()
    def observe(name):
        if name=='search' and not entered.is_set():
            entered.set()
            assert release.wait(5)
    case.fake.observe=observe
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(link,case)
        assert entered.wait(5)
        second=pool.submit(link,case).result(timeout=5)
        assert second.status=='journal_busy'
        release.set()
        assert first.result(timeout=5).status=='linked'
    assert case.fake.calls.count('create')==1
    with journal.LinkageJournal(case.root).transaction(case.record.linkage_key):
        with pytest.raises(journal.JournalError,match='journal_busy'):
            with journal.LinkageJournal(case.root).transaction(case.record.linkage_key):pass


def test_known_run_tag_changes_and_deleted_runs_are_never_replaced(case):
    assert link(case).status=='linked'
    case.fake.store['run-1']['tags']['apf.linkage_key']='changed'
    assert link(case).status=='conflict' and case.fake.calls.count('create')==1
    case.fake.store['run-1']['tags']=dict(saved(case).metadata_payload)
    case.fake.store['run-1']['lifecycle']='deleted'
    assert link(case).status=='linkage_uncertain' and case.fake.calls.count('create')==1


def test_journal_cas_and_creation_history_cannot_be_reset(case):
    case.fake.failure='create'
    assert link(case).status=='linkage_uncertain'
    with journal.LinkageJournal(case.root).transaction(case.record.linkage_key) as tx:
        old=tx.read()
        with pytest.raises(journal.JournalError,match='journal_revision_conflict'):
            tx.write(replace(old,revision=old.revision+1),expected_revision=old.revision+1)
        with pytest.raises(journal.JournalError,match='creation_history_cannot_be_reset'):
            tx.write(replace(old,status='pending',phase='intent',create_may_exist=False,revision=old.revision+1),expected_revision=old.revision)


def test_corrupt_unknown_journal_fields_are_not_replaced_or_echoed(case):
    assert link(case).status=='linked'
    path=case.root/(case.record.linkage_key+'.json')
    raw=json.loads(path.read_text());raw['api_token']='private-token-value-never-persist'
    path.write_text(json.dumps(raw));before=path.read_bytes();calls=len(case.fake.calls)
    result=link(case)
    assert result.status=='journal_failed' and path.read_bytes()==before and len(case.fake.calls)==calls
    assert 'private-token-value-never-persist' not in repr(result.diagnostics)


def test_competing_remote_clients_on_different_journals_detect_duplicates(case,tmp_path):
    initial_search=threading.Barrier(2)
    created=threading.Barrier(2)
    store={}
    shared_lock=threading.RLock()
    class Competitor(FakeClient):
        def search_runs(self,**kwargs):
            page=super().search_runs(**kwargs)
            if self.calls.count('search')==1:
                assert not page
                initial_search.wait(timeout=5)
            return page
        def create_run(self,**kwargs):
            run=super().create_run(**kwargs)
            created.wait(timeout=5)
            return run
    clients=[Competitor(store),Competitor(store)]
    for client in clients:client.lock=shared_lock
    roots=[(tmp_path/'first').resolve(),(tmp_path/'second').resolve()]
    for root in roots:root.mkdir()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(api.link_research_record,case.record,client=api.BoundMlflowClient(case.destination,client),
            destination=case.destination,journal_root=root,enabled=True) for client,root in zip(clients,roots)]
        results=[future.result(timeout=10) for future in futures]
    assert len(store)==2 and all(result.status=='duplicate' for result in results)
    assert all(client.calls.count('create')==1 for client in clients)


@pytest.mark.parametrize('after_commit',[False,True])
def test_interrupt_releases_local_lock_and_keeps_creating_phase(case,after_commit):
    original=case.fake.create_run
    def interrupted(**kwargs):
        if after_commit:original(**kwargs)
        raise KeyboardInterrupt('synthetic crash')
    case.fake.create_run=interrupted
    with pytest.raises(KeyboardInterrupt):link(case)
    assert saved(case).status=='creating' and saved(case).create_may_exist
    case.fake.create_run=original
    result=link(case)
    assert result.status==('linked' if after_commit else 'linkage_uncertain')
    assert len(case.fake.store)==(1 if after_commit else 0)


@pytest.mark.parametrize('fault',['missing_root','archive_root','symlink_entry','hardlink_entry','symlink_lock'])
def test_unsafe_or_missing_journal_locations_do_not_touch_clients(case,tmp_path,fault):
    c=case
    if fault=='missing_root':
        result=link(c,journal_root=tmp_path/'not-created')
        assert not (tmp_path/'not-created').exists()
    elif fault=='archive_root':
        root=tmp_path/('apf-research-bundle-v1-'+'0'*64)
        root.mkdir()
        result=link(c,journal_root=root)
        assert not list(root.iterdir())
    else:
        import os
        outside=tmp_path/'outside'
        outside.write_text('unchanged')
        suffix='.lock' if fault=='symlink_lock' else '.json'
        target=c.root/(c.record.linkage_key+suffix)
        if fault=='hardlink_entry':os.link(outside,target)
        else:target.symlink_to(outside)
        result=link(c)
        assert outside.read_text()=='unchanged'
    assert result.status=='journal_failed' and not c.fake.calls


def test_primary_journal_error_survives_failed_owned_stage_cleanup(case,monkeypatch):
    def fail(*args,**kwargs):raise OSError('private-token-value-never-persist')
    monkeypatch.setattr(journal.os,'fsync',fail)
    monkeypatch.setattr(journal.os,'unlink',fail)
    result=link(case)
    assert result.status=='journal_failed' and result.diagnostics==('journal_write_failed',)
    assert not case.fake.calls


def test_unrelated_remote_tags_are_not_persisted(case):
    seed(case)
    case.fake.store['existing']['tags']['password']='private-token-value-never-persist'
    result=link(case)
    assert result.status=='linked'
    assert 'private-token-value-never-persist' not in saved(case).canonical_json()


def test_pagination_cycle_is_an_explicit_search_failure(case):
    def cycle(**kwargs):return Page([],token='same-token')
    case.fake.search_runs=cycle
    result=link(case)
    assert result.status=='failed' and 'remote_search_failed' in result.diagnostics
    assert not case.fake.store


def test_scientific_identity_and_archive_are_never_changed_or_republished(case,monkeypatch):
    import builtins
    import socket
    import sqlite3
    from src.research import bundle
    from src.evaluation.execution.engine import ExecutionEngineV1
    before=case.record.canonical_json()
    original_import=builtins.__import__
    def forbidden(*args,**kwargs):pytest.fail('External operation outside the injected client is forbidden')
    def imports(name,*args,**kwargs):
        if name.split('.')[0] in {'mlflow','xgboost','sklearn','sqlalchemy','psycopg2','yfinance'}:forbidden()
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',imports)
    monkeypatch.setattr(socket,'create_connection',forbidden)
    monkeypatch.setattr(socket,'getaddrinfo',forbidden)
    monkeypatch.setattr(socket.socket,'connect',forbidden)
    monkeypatch.setattr(sqlite3,'connect',forbidden)
    monkeypatch.setattr(bundle,'publish_research_bundle',forbidden)
    monkeypatch.setattr(bundle,'open_research_bundle',forbidden)
    monkeypatch.setattr(ExecutionEngineV1,'run',forbidden)
    result=link(case)
    assert result.status=='linked' and case.record.canonical_json()==before
    assert result.intent.record_id==case.record.record_id and result.intent.linkage_key==case.record.linkage_key
