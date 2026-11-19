# Modeling and temporal evaluation

The supported model is a real three-class XGBoost classifier. The fixed synthetic example generates 180 weekday observations, records original source identities and explicitly declares UTC bar clocks. The five-observation forward-return target uses a 0.5% threshold: Down (0), Neutral (1), Up (2). This is not a five-calendar-day guarantee.

The candidate predictors are close, return_lag_1, ema_7, vol_30, volume_change and dayofweek. Lagged volatility and volume change use prior observations; the current close and EMA require the current bar to finish. Target construction uses future data solely for labels and records maturity separately from feature time. Input preparation preserves the permitted feature prefix and label tail.

After warm-up, frozen memberships are 60 fitting, five first-purge, 19 validation, five second-purge and 55 held-out rows. At both temporal boundaries, a label reaching the next protected block is excluded. These counts are specific to this fixture; future datasets require their own frozen protocol.

RF selection and RobustScaler fit only the fitting rows, then transform validation and held-out data. XGBoost settings in [the example](../examples/synthetic_research.py) are 24 estimators maximum, depth 2, learning rate 0.1, seed 42, one thread, histogram trees, multi:softprob, three classes, multiclass log loss and four-round early stopping. Validation labels enter early stopping and the information cutoff; they are not held out from model selection. Held-out evaluation records accuracy, log loss and a confusion matrix after prediction and signal construction.

Probability columns are mapped using estimator classes, not guessed. Predictions remain keyed to exact held-out identities. The execution signal mapping reads predicted classes, not realized labels. Feature availability is declared at close plus one minute and decisions at close plus two; model availability also includes validation/early-stopping information and declared latency. Actual fitting and generation times are recorded separately. Missing archived decision evidence remains incomplete; these declarations do not establish real historical point-in-time eligibility.

The objective is causal software validation, not competitive synthetic accuracy. Fixed seeds do not guarantee identical RF native state on independent refits; functional replay tests saved-state output reproduction rather than that stronger claim.
