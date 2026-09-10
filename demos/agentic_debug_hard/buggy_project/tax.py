"""Tax calculation logic."""

from .models import Order


# Tax rates by state
STATE_TAX_RATES = {
    "CA": 0.0725,   # California: 7.25%
    "TX": 0.0625,   # Texas: 6.25%
    "NY": 0.08,     # New York: 8%
    "WA": 0.065,    # Washington: 6.5%
    "OR": 0.0,      # Oregon: No sales tax
    "DEFAULT": 0.08,
}


def get_tax_rate(state: str) -> float:
    """Get the tax rate for a state."""
    return STATE_TAX_RATES.get(state, STATE_TAX_RATES["DEFAULT"])


def calculate_tax(amount: float, state: str = "DEFAULT") -> float:
    """Calculate tax on an amount."""
    rate = get_tax_rate(state)
    return round(amount * rate, 2)


def apply_state_tax(order: Order, state: str) -> None:
    """Apply state-specific tax rate to an order."""
    order.tax_rate = get_tax_rate(state)
