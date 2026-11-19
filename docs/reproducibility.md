# Reproducibility: distinct claims and recorded evidence

Reproducibility is evaluated at separate boundaries; success at one is not proof of all the others.

| Boundary | Supported evidence | Remaining limit |
|---|---|---|
| Source / artifact integrity | Frozen inventories, source hashes, receipt and bundle consistency | Hashes do not authenticate ownership or historical training |
| Saved-state functional replay | 55 synthetic predictions; maximum difference 0.0; zero class mismatches in recorded macOS Core run | Not independently reproduced training or universal cross-platform equivalence |
| Installation | P4-D2B fresh CPython 3.11.5 macOS arm64 Core install from 17 verified wheel inputs; pip check passed | Narrow existing artifact set; no complete bootstrap/native/build lock |
| Build | Original reviewed wheel retained | No independently repeated byte-identical build; no new v5 distribution |
| Decision chronology | Temporal manifest retains declared feature availability and model cutoffs | Complete per-prediction decisions are absent from the immutable V1 capture/bundle |
| Historical PIT | Synthetic observation and availability conventions are explicit | No real historical availability or model-existence attestation |
| Native Linux | Eight mandatory cases identified | Zero executed; runtime and dependency blockers |
| Financial performance | Deterministic accounting and cost checks | No real-market predictive or investment-performance evidence |

## Historical test results, not a combined total

| Stage | Candidate / platform | Recorded selection | Warnings |
|---|---|---|---:|
| P3-B | v2 / CPython 3.11.5 macOS arm64 | One process: 1,343 collected, 1,335 passed, eight native deselected; zero failures/errors/skips | 20 |
| P4-B | v3 / same platform | One process: 1,379 collected, 1,371 passed, eight native deselected; zero failures/errors/skips | 20 |
| P4-D1 | v4 / same platform | Targeted selection: 364 collected, 356 passed, eight native deselected; zero failures/errors/skips | 14 |
| P5-D1 | v6 / CPython 3.11.5 macOS arm64 | One process, 32 modules: 1,455 collected, 1,447 passed, eight native deselected; zero failures/errors/skips | 20 |
| P5-D1 targeted | v6 / same platform | Separate overlapping selection: 162 collected/passed; zero failures/errors/skips/deselections | 14 |
| P5-D3 | v7 / same platform | Targeted rights/metadata selection: 120 collected/passed; zero failures/errors/skips/deselections; 50 scientific files unchanged | 0 |
| P5-D4 | v8 / same platform | Targeted notice/metadata selection: 120 collected/passed; zero failures/errors/skips/deselections; approved notice finalized | 0 |
| P5-E | v8 / same platform | Bounded source-release checks: 40 collected/passed; zero failures/errors/skips/deselections; vulnerability status unverified | 0 |
| P6-A | Approved snapshot / same platform | Focused checks: 42 collected/passed; zero failures/errors/skips/deselections; 112 hashes and 50 scientific files unchanged | 0 |

The separate v3 guarded run overlapped the direct suite and had 21 warnings; it is not an additional unique-test total. Existing warnings were not suppressed. P4-D2B installation/demo/replay are stage checks, not pytest counts. P4-D2C is an evidence audit and P4-D3 is blocked Linux readiness, not passing scientific suites. The later targeted checks do not supersede the v6 consolidated scientific run. Counts overlap and must not be added together. None of these historical runs is described as a rerun on the current GitHub documentation commit.

The macOS developer command remains `python -m pytest tests -k 'not native_linux'` in the prepared full development environment. Core alone cannot run all tests. The independently executed v6 CI pilot rejected with `dependency_lock_incomplete_or_invalid` before its worker ran. It remains blocked by partial dependency locks, not a passing CI result. The manual workflows have not been executed as part of this documentation validation.

## Current reproducibility limits

Fixed seeds do not establish bit-identical RF native state across independent fits. Exact state remains preserved; no semantic identity normalization is introduced. The example's replay loads only its newly generated owned booster and is not a hostile-model sandbox.

Independent bundle verification retains expected incomplete decision chronology. Numerical replay is allowed to succeed within its narrower scope; never weaken an overall verification assertion merely to hide that limitation. Signals and execution CSVs are generated after publication and are not retroactively part of that archive.

macOS locks are LOCK_PARTIAL, Linux locks LOCK_UNAVAILABLE; vulnerability status and three dependency full-text license evidence gaps remain open. No native Linux, power-loss, complete build reproducibility, real historical PIT or real-market OOS claim is made. Source/licensing clearance remains independent of tests. See [limitations](limitations.md).

## Evidence provenance and interpretation

The summaries above were reconciled against the retained private stage reports:
P5D1_CI_VALIDATION.md, P5D3_RIGHTS_RESERVED_REVIEW.md,
P5D4_RIGHTS_FINALIZATION.md, P5E_FINAL_RELEASE_GATE.md and
P6A_PUBLICATION_SNAPSHOT.md. These names identify historical evidence, not files
shipped in this repository or publicly accessible links. No private machine path,
raw log or historical artifact is needed to read the summaries.

The v6 full session retained 20 warnings and explicitly classified 32 modules;
the separate targeted selection retained 14 warnings. V7, v8, P5-E and P6-A each
reported zero warnings in their own targeted sessions. A bounded source review
and a negative secret scan are not a vulnerability assessment. The unchanged
saved-state evidence remains 55 predictions, maximum probability difference 0.0
and zero class mismatches. Independent decision-time verification remains
incomplete. Neither numerical replay nor these test counts establish historical
PIT validity, real-market predictive performance or native Linux validation.
