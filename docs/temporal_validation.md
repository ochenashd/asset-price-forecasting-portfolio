# Temporal validation contract

All clocks in the artificial calendar are explicit UTC instants. Weekday sessions
are fixtures, not exchange holiday certification. A date label is never converted
into a real exchange closing time. Model features observe the synthetic 16:00
close; declared availability is 16:01; historical decisions are 16:02. Actual fit
completion and batch-generation clocks are recorded separately as current UTC.
A replay declaration is not evidence the model existed historically.

The frozen 180-row universe uses original source IDs r0..r179. Engineered rows
e31..e174 mature at their original source's fifth successor. The exact memberships
are fitting e31..e90, inner purge e91..e95, validation e96..e114, outer purge
e115..e119 and held-out e120..e174. No fraction-based resplitting occurs later.
Fitting target ends must be strictly before the first validation observation;
outer-training target ends must be strictly before the first held-out observation.

Causal EMA/lag/rolling features use only the chronological prefix. Label creation
alone reads future prices. All target and maturity fields are excluded from the
numeric predictor matrix. Correlation filtering and Random Forest importance use
only fitting rows; RobustScaler is fitted on the same rows. Validation labels
are used solely for early stopping and model selection availability. No parameter
search is performed on validation or held-out results.

The model-information cutoff includes fitting labels, validation labels, selection,
scaling and early-stopping state. Fitting and inference latencies are elapsed UTC
durations. Every decision must satisfy the existing strict information-cutoff rule
and the declared earliest availability. Held-out labels are read only for metrics,
after predictions and signals have been constructed. They are never adapter inputs.
