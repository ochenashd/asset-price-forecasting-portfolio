"""Cash/share accounting with decimal arithmetic and actual-notional fees."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from typing import Literal


def amount(value: float | Decimal) -> Decimal:
    """Convert an already validated numeric price or configuration value."""
    return Decimal(str(value))


def slipped_price(reference: float, rate: float, side: Literal["buy", "sell"]) -> Decimal:
    """Apply adverse slippage in decimal arithmetic to a validated reference price."""
    if side not in {"buy", "sell"}:
        raise ValueError("Slippage supports buy or sell fills only.")
    adjustment = amount(rate) if side == "buy" else -amount(rate)
    return amount(reference) * (1 + adjustment)


@dataclass
class Portfolio:
    """One fixed-quantity signed position; no simultaneous positions or financing.

    Unrealized P&L includes entry fees while a trade is open. On exit, the whole
    net trade P&L becomes realized. Thus equity always equals initial capital plus
    cumulative realized and current unrealized net P&L.
    """

    cash: Decimal
    quantity: int = 0
    entry_price: Decimal = Decimal(0)
    entry_fee: Decimal = Decimal(0)
    realized_pnl: Decimal = Decimal(0)
    gross_realized_pnl: Decimal = Decimal(0)
    realized_costs: Decimal = Decimal(0)
    entry_reference_price: Decimal = Decimal(0)
    entry_slippage_cost: Decimal = Decimal(0)
    slippage_costs: Decimal = Decimal(0)
    reference_gross_realized_pnl: Decimal = Decimal(0)
    restricted_proceeds: Decimal = Decimal(0)

    @property
    def unrestricted_cash(self) -> Decimal:
        """Restricted short proceeds are included in cash but cannot fund new risk."""
        return self.cash - self.restricted_proceeds

    def size_short_order(self, signal_close: float, fraction: float, fee_rate: float,
                         slippage_rate: float, max_exposure: float, initial_margin: float) -> int:
        """Freeze magnitude using post-cost equity and only the observed signal close."""
        if self.quantity or self.restricted_proceeds:
            raise ValueError("Short sizing requires a flat portfolio without restricted proceeds.")
        reference = amount(signal_close)
        executed = slipped_price(signal_close, slippage_rate, "sell")
        cost = reference - executed + executed * amount(fee_rate)
        cap = amount(max_exposure) * amount(fraction)
        by_exposure = cap * self.cash / (reference + cap * cost)
        by_margin = self.cash / (amount(initial_margin) * reference + cost)
        return int(min(by_exposure, by_margin).to_integral_value(rounding=ROUND_FLOOR))

    def short_entry_error(self, quantity: int, reference: float, executed: Decimal,
                          fee_rate: float, max_exposure: float, initial_margin: float) -> str | None:
        """Check frozen magnitude at the actual reference open, including entry costs."""
        if self.quantity or self.restricted_proceeds or quantity <= 0:
            return "short_entry_requires_flat_portfolio"
        fee = quantity * executed * amount(fee_rate)
        liability = quantity * amount(reference)
        equity = self.cash + quantity * executed - fee - liability
        if fee > self.unrestricted_cash or equity <= 0:
            return "insufficient_short_collateral"
        if liability > amount(max_exposure) * equity:
            return "short_exposure_cap"
        if equity < amount(initial_margin) * liability:
            return "insufficient_initial_margin"
        return None

    def enter_short(self, quantity: int, price: Decimal, fee_rate: float, *,
                    reference_price: float, max_exposure: float, initial_margin: float) -> Decimal:
        """Sell borrowed shares and restrict gross proceeds; debit commission once."""
        error = self.short_entry_error(quantity, reference_price, price, fee_rate,
                                       max_exposure, initial_margin)
        if error is not None:
            raise ValueError(error)
        slippage = quantity * (amount(reference_price) - price)
        if slippage < 0:
            raise ValueError("A short sale cannot improve on its reference price.")
        notional = quantity * price
        fee = notional * amount(fee_rate)
        self.cash += notional - fee
        self.restricted_proceeds = notional
        self.quantity = -quantity
        self.entry_price = price
        self.entry_fee = fee
        self.realized_costs += fee
        self.entry_reference_price = amount(reference_price)
        self.entry_slippage_cost = slippage
        self.slippage_costs += slippage
        return fee

    def margin_values(self, reference: float, initial_margin: float,
                      maintenance_margin: float) -> dict[str, Decimal]:
        """Report marked liability and collateral; restrictions are not expenses.

        Available collateral uses initial margin, while margin_excess uses the
        maintenance requirement. A negative excess is never clamped away.
        """
        liability = max(0, -self.quantity) * amount(reference)
        equity = self.equity(reference)
        initial = liability * amount(initial_margin)
        maintenance = liability * amount(maintenance_margin)
        return {"restricted_proceeds": self.restricted_proceeds,
                "unrestricted_cash": self.unrestricted_cash, "short_liability": liability,
                "initial_margin_requirement": initial, "maintenance_requirement": maintenance,
                "margin_excess": equity - maintenance,
                "available_collateral": max(Decimal(0), min(self.unrestricted_cash, equity - initial))}

    def estimated_flat_cash(self, reference: float, fee_rate: float,
                            slippage_rate: float) -> Decimal:
        """Estimate post-exit cash without changing cash, shares or restrictions.

        Only the observed decision close is supplied. Actual execution still
        requires a fill before collateral is released or new risk is accepted.
        """
        if not self.quantity:
            return self.cash
        side = "buy" if self.quantity < 0 else "sell"
        executed = slipped_price(reference, slippage_rate, side)
        return self.cash + self.quantity * executed - abs(self.quantity) * executed * amount(fee_rate)

    def size_order(self, signal_close: float, fraction: float, fee_rate: float,
                   slippage_rate: float = 0.0) -> int:
        """Freeze whole-share quantity using only cash and the observed signal close."""
        unit_cost = slipped_price(signal_close, slippage_rate, "buy") * (1 + amount(fee_rate))
        return int((self.cash * amount(fraction) / unit_cost).to_integral_value(rounding=ROUND_FLOOR))

    def can_afford(self, quantity: int, price: float | Decimal, fee_rate: float) -> bool:
        """Check the actual fill notional plus fee without resizing the order."""
        return quantity > 0 and amount(price) * quantity * (1 + amount(fee_rate)) <= self.cash

    def enter(self, quantity: int, price: float | Decimal, fee_rate: float, *,
              reference_price: float | Decimal | None = None) -> Decimal:
        """Debit shares and fees at the observed open; return the charged fee."""
        if self.quantity or not self.can_afford(quantity, price, fee_rate):
            raise ValueError("Entry requires a flat portfolio and an affordable positive quantity.")
        notional = amount(price) * quantity
        fee = notional * amount(fee_rate)
        reference = amount(price if reference_price is None else reference_price)
        slippage = quantity * (amount(price) - reference)
        if slippage < 0:
            raise ValueError("A buy fill must not improve on its reference price.")
        self.cash -= notional + fee
        self.quantity = quantity
        self.entry_price = amount(price)
        self.entry_fee = fee
        self.realized_costs += fee
        self.entry_reference_price = reference
        self.entry_slippage_cost = slippage
        self.slippage_costs += slippage
        return fee

    def exit(self, price: float | Decimal, fee_rate: float, *,
             reference_price: float | Decimal | None = None) -> tuple[Decimal, Decimal]:
        """Close at an actual modeled fill and realize the net trade P&L."""
        if self.quantity == 0:
            raise ValueError("Exit requires an active position.")
        notional = amount(price) * abs(self.quantity)
        fee = notional * amount(fee_rate)
        reference = amount(price if reference_price is None else reference_price)
        slippage = self.quantity * (reference - amount(price))
        if slippage < 0:
            raise ValueError("An exit fill must not improve on its reference price.")
        gross = self.quantity * (amount(price) - self.entry_price)
        realized = gross - self.entry_fee - fee
        self.cash += self.quantity * amount(price) - fee
        self.realized_pnl += realized
        self.gross_realized_pnl += gross
        self.realized_costs += fee
        self.slippage_costs += slippage
        self.reference_gross_realized_pnl += self.quantity * (reference - self.entry_reference_price)
        self.quantity = 0
        self.entry_price = Decimal(0)
        self.entry_fee = Decimal(0)
        self.entry_reference_price = Decimal(0)
        self.entry_slippage_cost = Decimal(0)
        self.restricted_proceeds = Decimal(0)
        return fee, realized

    def equity(self, close: float) -> Decimal:
        """Mark signed position value plus cash at the current session close."""
        return self.cash + self.quantity * amount(close)

    def unrealized(self, close: float) -> Decimal:
        """Preserve the Stage 1 net mark, including the open entry commission."""
        return self.gross_unrealized(close) - self.entry_fee

    def gross_unrealized(self, close: float) -> Decimal:
        """Return open mark-to-market P&L with incurred fees tracked separately."""
        return self.quantity * (amount(close) - self.entry_price)

    @property
    def net_realized_pnl(self) -> Decimal:
        """Recognize gross exits and all fees when their fills actually occur."""
        return self.gross_realized_pnl - self.realized_costs

    def reference_unrealized(self, close: float) -> Decimal:
        """Mark against the reference entry, with incurred slippage tracked separately."""
        return self.quantity * (amount(close) - self.entry_reference_price)

    @property
    def reference_net_realized_pnl(self) -> Decimal:
        """Attribute commissions and slippage to their fill dates without cash debits."""
        return self.reference_gross_realized_pnl - self.realized_costs - self.slippage_costs
