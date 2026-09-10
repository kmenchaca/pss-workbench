"""Discount calculation logic."""

from typing import Optional
from .models import Order


# Discount tiers based on subtotal
DISCOUNT_TIERS = [
    (500.0, 15.0),   # Orders >= $500 get 15% off
    (200.0, 10.0),   # Orders >= $200 get 10% off
    (100.0, 5.0),    # Orders >= $100 get 5% off
]


def calculate_tier_discount(subtotal: float) -> float:
    """Calculate discount percentage based on order subtotal."""
    for threshold, discount in DISCOUNT_TIERS:
        if subtotal >= threshold:
            return discount
    return 0.0


def apply_discount_code(order: Order, code: str) -> Optional[str]:
    """Apply a discount code to an order.

    Returns error message if code is invalid, None on success.
    """
    codes = {
        "SAVE10": 10.0,
        "SAVE20": 20.0,
        "WELCOME": 15.0,
    }

    if code not in codes:
        return f"Invalid discount code: {code}"

    # Discount codes override tier discounts if better
    code_discount = codes[code]
    if code_discount > order.discount_percent:
        order.discount_percent = code_discount

    return None


def calculate_final_discount(order: Order) -> float:
    """Calculate the final discount for an order.

    Applies tier discount if no code discount is better.
    """
    tier_discount = calculate_tier_discount(order.subtotal)

    # Use the better of tier vs existing discount
    if tier_discount > order.discount_percent:
        return tier_discount
    return order.discount_percent
