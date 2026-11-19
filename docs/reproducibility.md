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

The separate v3 guarded run overlapped the direct suite and had 21 warnings; it is not an additional unique-test total. Existing warnings were not suppressed. P4-D2B installation/demo/replay are stage checks, not pytest counts. P4-D2C is an evidence audit and P4-D3 is blocked Linux readiness, not passing scientific suites. V5 documentation-only validation is recorded in its separate private report; it does not supersede those scientific runs.

The macOS developer command remains `python -m pytest tests -k 'not native_linux'` in the prepared full development environment. Core alone cannot run all tests. Manual CI is intentionally blocked by incomplete dependency evidence. No GitHub workflow has run for this release validation.

## Current reproducibility limits

Fixed seeds do not establish bit-identical RF native state across independent fits. Exact state remains preserved; no semantic identity normalization is introduced. The example's replay loads only its newly generated owned booster and is not a hostile-model sandbox.

Independent bundle verification retains expected incomplete decision chronology. Numerical replay is allowed to succeed within its narrower scope; never weaken an overall verification assertion merely to hide that limitation. Signals and execution CSVs are generated after publication and are not retroactively part of that archive.

macOS locks are LOCK_PARTIAL, Linux locks LOCK_UNAVAILABLE; vulnerability status and three dependency full-text license evidence gaps remain open. No native Linux, power-loss, complete build reproducibility, real historical PIT or real-market OOS claim is made. Source/licensing clearance remains independent of tests. See [limitations](limitations.md).
