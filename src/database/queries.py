INSERT_ASSET = """
INSERT INTO assets (
    symbol,
    asset_name,
    asset_type,
    exchange,
    quote_currency
)
VALUES (
    :symbol,
    :asset_name,
    :asset_type,
    :exchange,
    :quote_currency
)
ON CONFLICT (symbol)
DO UPDATE SET
    asset_name = EXCLUDED.asset_name,
    asset_type = EXCLUDED.asset_type,
    exchange = EXCLUDED.exchange,
    quote_currency = EXCLUDED.quote_currency
RETURNING asset_id;
"""


GET_ASSET_BY_SYMBOL = """
SELECT
    asset_id,
    symbol,
    asset_name,
    asset_type,
    exchange,
    quote_currency,
    created_at
FROM assets
WHERE symbol = :symbol;
"""


INSERT_MARKET_PRICE = """
INSERT INTO market_prices (
    asset_id,
    timestamp,
    timeframe,
    open,
    high,
    low,
    close,
    volume,
    source
)
VALUES (
    :asset_id,
    :timestamp,
    :timeframe,
    :open,
    :high,
    :low,
    :close,
    :volume,
    :source
)
ON CONFLICT (asset_id, timestamp, timeframe)
DO UPDATE SET
    open = EXCLUDED.open,
    high = EXCLUDED.high,
    low = EXCLUDED.low,
    close = EXCLUDED.close,
    volume = EXCLUDED.volume,
    source = EXCLUDED.source;
"""


GET_MARKET_PRICES = """
SELECT
    p.timestamp,
    p.open,
    p.high,
    p.low,
    p.close,
    p.volume,
    p.timeframe,
    p.source
FROM market_prices p
JOIN assets a ON p.asset_id = a.asset_id
WHERE a.symbol = :symbol
  AND p.timeframe = :timeframe
  AND (:start_date IS NULL OR p.timestamp >= :start_date)
  AND (:end_date IS NULL OR p.timestamp <= :end_date)
ORDER BY p.timestamp;
"""


INSERT_MODEL_RUN = """
INSERT INTO model_runs (
    model_name,
    asset_id,
    target_name,
    timeframe,
    train_start,
    train_end,
    test_start,
    test_end,
    parameters_json,
    metrics_json,
    mlflow_run_id
)
VALUES (
    :model_name,
    :asset_id,
    :target_name,
    :timeframe,
    :train_start,
    :train_end,
    :test_start,
    :test_end,
    CAST(:parameters_json AS JSONB),
    CAST(:metrics_json AS JSONB),
    :mlflow_run_id
)
RETURNING run_id;
"""


INSERT_PREDICTION = """
INSERT INTO model_predictions (
    run_id,
    asset_id,
    timestamp,
    y_true,
    y_pred,
    y_true_class,
    predicted_class,
    proba_down,
    proba_neutral,
    proba_up
)
VALUES (
    :run_id,
    :asset_id,
    :timestamp,
    :y_true,
    :y_pred,
    :y_true_class,
    :predicted_class,
    :proba_down,
    :proba_neutral,
    :proba_up
)
ON CONFLICT (run_id, asset_id, timestamp)
DO UPDATE SET
    y_true = EXCLUDED.y_true,
    y_pred = EXCLUDED.y_pred,
    y_true_class = EXCLUDED.y_true_class,
    predicted_class = EXCLUDED.predicted_class,
    proba_down = EXCLUDED.proba_down,
    proba_neutral = EXCLUDED.proba_neutral,
    proba_up = EXCLUDED.proba_up;
"""

INSERT_BACKTEST_RESULT = """
INSERT INTO backtest_results (
    run_id,
    asset_id,
    strategy_name,
    initial_capital,
    final_capital,
    total_return,
    annualized_return,
    benchmark_return,
    strategy_return,
    volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    win_rate,
    number_of_trades,
    number_of_rebalances,
    long_only,
    exposure,
    long_signals,
    short_signals,
    flat_signals
)
VALUES (
    :run_id,
    :asset_id,
    :strategy_name,
    :initial_capital,
    :final_capital,
    :total_return,
    :annualized_return,
    :benchmark_return,
    :strategy_return,
    :volatility,
    :sharpe_ratio,
    :sortino_ratio,
    :max_drawdown,
    :win_rate,
    :number_of_trades,
    :number_of_rebalances,
    :long_only,
    :exposure,
    :long_signals,
    :short_signals,
    :flat_signals
)
RETURNING backtest_id;
"""


GET_MODEL_LEADERBOARD = """
SELECT
    r.model_name,
    a.symbol,
    r.timeframe,
    b.strategy_name,
    b.total_return,
    b.benchmark_return,
    b.strategy_return,
    b.sharpe_ratio,
    b.sortino_ratio,
    b.max_drawdown,
    b.win_rate,
    b.number_of_trades,
    b.number_of_rebalances,
    b.long_only,
    b.exposure,
    b.long_signals,
    b.short_signals,
    b.flat_signals,
    b.created_at
FROM backtest_results b
JOIN model_runs r ON b.run_id = r.run_id
JOIN assets a ON b.asset_id = a.asset_id
ORDER BY b.sharpe_ratio DESC NULLS LAST;
"""

INSERT_TECHNICAL_FEATURE = """
INSERT INTO technical_features (
    asset_id,
    timestamp,
    timeframe,

    return_1d,
    return_7d,
    return_14d,
    return_30d,

    volatility_7d,
    volatility_30d,

    rsi_14,
    macd,
    macd_signal,
    macd_hist,

    bb_upper,
    bb_middle,
    bb_lower,
    bb_width,

    sma_7,
    sma_14,
    sma_30,
    sma_60,

    ema_7,
    ema_14,
    ema_30,
    ema_60,

    volume_change,

    target,
    target_return,
    features_json,
    updated_at
)
VALUES (
    :asset_id,
    :timestamp,
    :timeframe,

    :return_1d,
    :return_7d,
    :return_14d,
    :return_30d,

    :volatility_7d,
    :volatility_30d,

    :rsi_14,
    :macd,
    :macd_signal,
    :macd_hist,

    :bb_upper,
    :bb_middle,
    :bb_lower,
    :bb_width,

    :sma_7,
    :sma_14,
    :sma_30,
    :sma_60,

    :ema_7,
    :ema_14,
    :ema_30,
    :ema_60,

    :volume_change,

    :target,
    :target_return,
    CAST(:features_json AS JSONB),
    CURRENT_TIMESTAMP
)
ON CONFLICT (asset_id, timestamp, timeframe)
DO UPDATE SET
    return_1d = EXCLUDED.return_1d,
    return_7d = EXCLUDED.return_7d,
    return_14d = EXCLUDED.return_14d,
    return_30d = EXCLUDED.return_30d,

    volatility_7d = EXCLUDED.volatility_7d,
    volatility_30d = EXCLUDED.volatility_30d,

    rsi_14 = EXCLUDED.rsi_14,
    macd = EXCLUDED.macd,
    macd_signal = EXCLUDED.macd_signal,
    macd_hist = EXCLUDED.macd_hist,

    bb_upper = EXCLUDED.bb_upper,
    bb_middle = EXCLUDED.bb_middle,
    bb_lower = EXCLUDED.bb_lower,
    bb_width = EXCLUDED.bb_width,

    sma_7 = EXCLUDED.sma_7,
    sma_14 = EXCLUDED.sma_14,
    sma_30 = EXCLUDED.sma_30,
    sma_60 = EXCLUDED.sma_60,

    ema_7 = EXCLUDED.ema_7,
    ema_14 = EXCLUDED.ema_14,
    ema_30 = EXCLUDED.ema_30,
    ema_60 = EXCLUDED.ema_60,

    volume_change = EXCLUDED.volume_change,

    target = EXCLUDED.target,
    target_return = EXCLUDED.target_return,
    features_json = EXCLUDED.features_json,
    updated_at = CURRENT_TIMESTAMP;
"""


GET_TECHNICAL_FEATURES = """
SELECT
    f.timestamp,
    f.timeframe,
    f.return_1d,
    f.return_7d,
    f.return_14d,
    f.return_30d,
    f.volatility_7d,
    f.volatility_30d,
    f.rsi_14,
    f.macd,
    f.macd_signal,
    f.macd_hist,
    f.bb_upper,
    f.bb_middle,
    f.bb_lower,
    f.bb_width,
    f.sma_7,
    f.sma_14,
    f.sma_30,
    f.sma_60,
    f.ema_7,
    f.ema_14,
    f.ema_30,
    f.ema_60,
    f.volume_change,
    f.target,
    f.target_return,
    f.features_json
FROM technical_features f
JOIN assets a ON f.asset_id = a.asset_id
WHERE a.symbol = :symbol
  AND f.timeframe = :timeframe
  AND (:start_date IS NULL OR f.timestamp >= :start_date)
  AND (:end_date IS NULL OR f.timestamp <= :end_date)
ORDER BY f.timestamp;
"""

INSERT_TRADING_SIGNAL = """
INSERT INTO trading_signals (
    prediction_id,
    run_id,
    asset_id,
    timestamp,
    strategy_name,
    signal,
    position,
    predicted_class,
    proba_down,
    proba_neutral,
    proba_up,
    target_return,
    min_up_proba,
    min_down_proba,
    up_threshold,
    down_threshold,
    threshold_method
)
SELECT
    p.prediction_id,
    :run_id,
    :asset_id,
    :timestamp,
    :strategy_name,
    :signal,
    :position,
    :predicted_class,
    :proba_down,
    :proba_neutral,
    :proba_up,
    :target_return,
    :min_up_proba,
    :min_down_proba,
    :up_threshold,
    :down_threshold,
    :threshold_method
FROM model_predictions p
WHERE p.run_id = :run_id
  AND p.asset_id = :asset_id
  AND p.timestamp = :timestamp
ON CONFLICT (run_id, asset_id, timestamp, strategy_name)
DO UPDATE SET
    prediction_id = EXCLUDED.prediction_id,
    signal = EXCLUDED.signal,
    position = EXCLUDED.position,
    predicted_class = EXCLUDED.predicted_class,
    proba_down = EXCLUDED.proba_down,
    proba_neutral = EXCLUDED.proba_neutral,
    proba_up = EXCLUDED.proba_up,
    target_return = EXCLUDED.target_return,
    min_up_proba = EXCLUDED.min_up_proba,
    min_down_proba = EXCLUDED.min_down_proba,
    up_threshold = EXCLUDED.up_threshold,
    down_threshold = EXCLUDED.down_threshold,
    threshold_method = EXCLUDED.threshold_method;
"""
