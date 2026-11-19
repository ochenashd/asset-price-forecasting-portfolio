# Architecture and supported scope

The supported application is [the synthetic example](../examples/synthetic_research.py), not the original forecasting pipeline. It assembles existing contracts directly and performs no provider ingestion. The README's Mermaid diagram is a dataflow summary of this route.

| Layer | Implemented responsibility | Supported evidence boundary |
|---|---|---|
| Generator | Deterministic modeling OHLCV and independently generated execution prices | Artificial weekday calendar and UTC clocks; no authenticated market source |
| Features / input preparation | Causal indicators, original row/maturity identity and frozen partitions | Labels use an authorized future maturity tail; predictors do not |
| Fit | Fitting-only RF selection and RobustScaler; real XGBoost with mature validation early stopping | Held-out labels excluded from fitting and selection |
| Capture / receipt | Keyed predictions, explicit estimator classes, exact fitted-state and source bindings | Actual batch time distinct from declared historical availability |
| Local bundle | Non-overwriting publication of owned artifacts and independent integrity checks | Complete archived prediction decision timing is absent |
| Replay | Reconstruct inputs and reproduce saved model outputs | Trusted fresh model only; not independent training reproduction |
| Execution | Direct Signal/MarketBar construction, next-open fills and daily reporting | After bundle publication; CSVs outside the research bundle |
| Optional tracking | Fresh local SQLite linkage to an accepted bundle | Explicit opt-in; no remote tracking or historical store |

`src/research` owns scientific artifact lifecycle. `src/evaluation/execution` owns execution contracts, simulation and performance reporting. `src/tracking` owns durable linkage and reconciliation. None replaces a verified market-data provenance source.

The independent verifier separates integrity, scientific consistency, runtime loading and functional reproduction. Verified archive bytes do not imply verified decision chronology. The example's replay path reports numerical reproduction without promoting incomplete temporal evidence.

## Implemented versus demonstrated

The Core installed demonstration and the recorded macOS regression runs support the documented synthetic workflow. Shorting, restricted proceeds, margin outcomes, replacement and terminal policies have engine-level synthetic tests, but the example uses only Long/Flat. Market snapshot/eligibility modules exist as fail-closed evidence contracts; the example bypasses that adapter path with explicitly synthetic bars and does not authenticate data.

Limited legacy model/database/backtest files remain for compatibility, source bindings and regression guards. They are not invoked by the supported demonstration. No claim is made for a validated neural, statistical-model or LightGBM portfolio workflow. Native Linux and genuinely historical market-data research remain unvalidated extensions.
