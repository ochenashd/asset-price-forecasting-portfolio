# Execution and tracking methodology

The demo compares Long/Flat fixed_horizon and replace_next_open policies. It maps
class 2 to Long and classes 0/1 to explicit Flat. Each policy receives the same
keyed capture, independent synthetic market feed, declared sessions and endpoint.
An order cannot execute before the next session opening. Five-session exits are
pre-scheduled close fills, never exit decisions based on that closing price.

Whole-share sizing uses information available at the preceding decision; actual
opening affordability is checked at fill. Initial capital is 10,000, allocation
is 50%, commission is 10 bps per filled side and adverse slippage is 5 bps per
side. Slippage affects cash through executed prices exactly once. Final liquidation
is predeclared at session-179. Daily equity equals cash plus marked share value.
Gross exposure, completed-trade results and turnover are distinct concepts.

Sharpe uses arithmetic session excess returns with population standard deviation;
Sortino uses full-sample downside deviation. Effective annual risk-free rate is 2%
and declared sampling is 252 sessions/year. CAGR uses actual elapsed time and
365.25 days/year. Initial capital participates in drawdown. Undefined ratios remain
undefined (JSON null), never epsilon-based artificial profitability. Daily close
marks cannot measure intraday drawdown. Synthetic business-day sampling is not an
independently verified market calendar. No borrow, funding, dividends or corporate
actions are simulated by this Long-only example. No benchmark is implemented.

Tracking is OFF by default. `--track-local` creates a fresh owned SQLite backend,
artifact root and durable linkage journal. Accepted bundle metadata is linked via
an explicitly bound MLflow client, then inspected by read-only reconciliation.
No global MLflow destination, external connection, automatic remote create retry
or production store is used. Tracking status cannot upgrade PIT or execution
certification. An immutable local bundle is not an authenticated external dataset.
