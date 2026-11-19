# Test coverage and historical validation

The current full-suite coverage registry contains 32 approved modules. It adds
tests/test_portfolio_documentation.py (16 portable static cases) to the retained
31-module v4 contract. CI contract regressions check the actual checkout, reject
unknown modules and preserve all eight mandatory native Linux cases. The driver
still selects the whole tests directory; macOS alone excludes the eight known
native cases. Skips, missing coverage and unsupported exclusions remain failures.

Use the reviewed development environment and one pytest process:
`python -m pytest tests -k 'not native_linux'`. Core alone is insufficient for the
full tests. Optional local MLflow fixtures remain function-scoped and isolated.
The full lock-enforced pilot is still intentionally blocked by partial macOS and
unavailable Linux locks. No native Linux result follows from metadata tests.

The v6 consolidated selection executed 1,455 collected cases in one CPython
3.11.5 macOS arm64 process: **1,447 passed**, eight native Linux tests deselected,
zero failures/errors/skips and 20 retained warnings. All 32 approved modules were
represented. A separate targeted selection passed 162 cases with 14 warnings;
these overlap the full session and are not added to its total. The independent
lock-enforced CI pilot was blocked before worker execution, not passed.

Later targeted sessions: v7 120 passed, v8 120 passed, P5-E 40 passed and P6-A
42 passed, each with zero failures or warnings. P6-A also checked all 112
approved paths and all 50 scientific hashes. These are separate historical
validations, not a full-suite rerun on the current GitHub commit. See
[reproducibility](reproducibility.md) for scope and evidence references. Historical 30/31-module counts
and private per-case evidence references describe earlier candidates, not current
CI acceptance. Do not add overlapping run totals or imply unavailable evidence is
included in a public source snapshot. Decision chronology, rights and vulnerability
limitations remain unchanged.

---

# Candidate v4 test matrix extension

The 30 v3 modules below remain mandatory. V4 adds tests/test_portfolio_lock_contract.py,
making 31 modules in the full driver. New cases are portable offline schema/hash/
profile/CI/documentation tests with synthetic bytes and temporary roots; they add
no fitting, inference, scientific execution or new runtime dependencies. Requirements
are stdlib + existing pytest/PyYAML. Hash tests do not install their synthetic files.

The following v3 per-case totals/timings are preserved historical evidence. V4
executed counts are recorded separately in P4D1_VALIDATION.md. The lock-enforced
CI gate currently rejects, so no new full CI success or native Linux run is implied.

---

## Preserved v3 coverage matrix

# Candidate v3 test coverage

Historical v3 rights-review status was **BLOCKED_BY_IP_OR_LICENSE**; this records
that stage, not current repository visibility or a new rights determination.
Counts below are collected cases,
not independently certified financial outcomes. The full per-case JSON/CSV matrix
is retained outside the source tree in private validation evidence. It records
node identity, category, platform, direct dependency imports, fixture dependencies,
filesystem effects, real-fitting requirements, observed result and JUnit duration.

## Coverage and executed platform

The v3 collection has 1,379 cases in 30 modules. The macOS consolidated command
selected and passed 1,371 in ONE pytest session, with eight Linux-native cases
deselected, zero failures/errors/skips, and 20 unsuppressed warnings. The separate
guarded CI driver also ran one full worker session; its counts overlap, not add.
No native Linux case has executed. Metadata-only Linux acceptance unit tests are
not Linux runtime evidence. CI rejects an unexpected platform or simulated Linux.

| Module | Collected | Linux-only | Cases using real-fit paths | Category |
| --- | ---: | ---: | ---: | --- |
| test_database_credential_security.py | 16 | 0 | 0 | Portable credential/configuration unit tests; no database connection |
| test_execution_adapter.py | 88 | 0 | 0 | Temporal/identity/probability metadata contracts |
| test_execution_engine_v1.py | 42 | 0 | 0 | Deterministic synthetic execution/accounting and performance |
| test_execution_metrics.py | 64 | 0 | 0 | Deterministic synthetic execution/accounting and performance |
| test_execution_model_provenance.py | 88 | 0 | 0 | Temporal/identity/probability metadata contracts |
| test_execution_prediction_export.py | 132 | 0 | 0 | Temporal/identity/probability metadata contracts |
| test_execution_reversals.py | 47 | 0 | 0 | Deterministic synthetic execution/accounting and performance |
| test_execution_short.py | 41 | 0 | 0 | Deterministic synthetic execution/accounting and performance |
| test_execution_slippage.py | 22 | 0 | 0 | Deterministic synthetic execution/accounting and performance |
| test_execution_terminal_policy.py | 59 | 0 | 0 | Deterministic synthetic execution/accounting and performance |
| test_execution_trade_ledger.py | 19 | 0 | 0 | Deterministic synthetic execution/accounting and performance |
| test_market_eligibility.py | 87 | 0 | 0 | Synthetic evidence/eligibility contracts; no provider certification |
| test_market_snapshot.py | 52 | 0 | 0 | Immutable synthetic file snapshot and hash validation |
| test_mlflow_fixture_isolation.py | 15 | 0 | 0 | MLflow fixture lifecycle and no-extra-fit guards |
| test_portfolio_ci_contract.py | 36 | 0 | 0 | CI coverage/security contracts; synthetic metadata, not native Linux |
| test_preprocessing_causality.py | 11 | 0 | 0 | Causal feature-input preprocessing controls |
| test_public_source_manifest.py | 2 | 0 | 0 | Exact source identity and package resources |
| test_research_bundle.py | 87 | 0 | 0 | Immutable local bundle publication and filesystem controls |
| test_research_bundle_verification.py | 56 | 0 | 0 | Data-only receipt/state/bundle verification and byte-only CLI |
| test_research_filesystem_ci_contract.py | 14 | 0 | 0 | Static filesystem-only CI coverage and safety |
| test_research_filesystem_contracts.py | 17 | 0 | 0 | Static filesystem source contracts |
| test_research_filesystem_crash_safety.py | 36 | 0 | 0 | Portable POSIX fault injection and owned subprocess termination |
| test_research_filesystem_linux_runtime.py | 18 | 8 | 0 | Portable preflight plus eight separately gated Linux-native cases |
| test_research_input_adapter.py | 44 | 0 | 0 | Frozen original identities, causal features and two maturity purges |
| test_research_mlflow_linkage.py | 53 | 0 | 0 | Fake-client linkage/journal fault recovery; no real MLflow server |
| test_research_mlflow_local_backend.py | 33 | 0 | 0 | Real new local SQLite MLflow integration, no model fitting |
| test_research_protocol.py | 36 | 0 | 0 | Frozen protocol/partition and artifact identity contracts |
| test_research_tracking_reconciliation.py | 51 | 0 | 0 | Read-only reconciliation over synthetic archives/journals/clients |
| test_research_tracking_record.py | 102 | 0 | 0 | Data-only scientific receipt/tracking records |
| test_synthetic_research.py | 11 | 0 | 4 | Synthetic real-XGBoost end-to-end, receipts/replay and causal input checks |

## Dependencies, runtime and side effects

Use the complete declared core + tracking + pytest profile for the consolidated
suite. Direct test imports in the private matrix are static evidence, not a
complete minimal dependency closure: imported helpers and fixtures add dependencies.
The five modules selected by scripts/research_filesystem_ci.py remain a separate
211-case subset using pytest, NumPy, pandas, PyArrow and OmegaConf. It does not
install XGBoost, scikit-learn or MLflow. Its eight native cases are mandatory on
Linux; do not add the overlapping job's counts to the research total.

The real-fitting cases are confined to existing synthetic example tests. Two cases
share one completed-demo module fixture; the authorized-row test runs its existing
fit and the label-invariance test runs its existing two fits. No new test or CI
driver adds fitting/inference algorithms. MLflow tests create only new temporary
local SQLite/artifact roots; lifecycle guards reject network, historical backend,
training/inference and fluent logging operations where required. Execution tests
use deterministic synthetic prices, with no provider or real-market backtest.

Metadata/unit tests are portable subject to the declared dependencies. POSIX
filesystem tests use owned temporary files, fault injection and subprocesses;
eight separate native Linux cases require real renameat2/fsync/flock/process-kill
behavior. No crash simulation proves power-loss durability. Per-case elapsed
seconds come from the executed portable JUnit: shared fixture cost is attributed
to the first requesting case, not a per-test performance guarantee. Native cases
have no measured duration. Overall CI limits are authoritative time budgets.

## Fail-closed CI acceptance

Both workflows are manual-only. At the historical v3 stage, the full driver
required all 30 classified modules; the current registry requires all 32 modules.
The preserved acceptance contract requires exact eight-native collection, mandatory source-identity,
isolation and real integration cases, nonzero selection, every selected case
passed and matching clean JUnit. Linux allows no deselection; macOS permits ONLY
the eight known native exclusions. Skips, xfails, errors and omissions reject.
Future modules must be explicitly classified before the gate accepts them.
Python 3.11.5, real host checks, owned empty temporary roots and matching installed
scientific source imports are preconditions. This is not a dependency lock,
hostile-code sandbox, rights clearance or proof of Linux execution.
