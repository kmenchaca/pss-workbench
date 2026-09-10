"""Order processing logic."""

import uuid
from typing import Optional
from .models import Order, OrderItem, OrderStatus
from .catalog import get_product
from .discounts import calculate_final_discount
from .tax import apply_state_tax


def create_order(customer_id: str, state: str = "DEFAULT") -> Order:
    """Create a new order for a customer."""
    order = Order(
        id=str(uuid.uuid4())[:8],
        customer_id=customer_id,
    )
    apply_state_tax(order, state)
    return order


def add_item_to_order(
    order: Order,
    product_id: str,
    quantity: int = 1,
) -> Optional[str]:
    """Add an item to an order.

    Returns error message if product not found, None on success.
    """
    product = get_product(product_id)
    if product is None:
        return f"Product not found: {product_id}"

    if not product.in_stock:
        return f"Product out of stock: {product_id}"

    # Check if item already exists in order
    for item in order.items:
        if item.product_id == product_id:
            item.quantity += quantity
            return None

    # Add new item
    item = OrderItem(
        product_id=product_id,
        quantity=quantity,
        unit_price=product.price,
    )
    order.items.append(item)
    return None


def finalize_order(order: Order) -> Optional[str]:
    """Finalize an order, applying discounts.

    Returns error message if order is invalid, None on success.
    """
    if not order.items:
        return "Cannot finalize empty order"

    if order.status != OrderStatus.PENDING:
        return f"Order already {order.status.value}"

    # Apply best discount
    order.discount_percent = calculate_final_discount(order)
    order.status = OrderStatus.CONFIRMED

    return None


def get_order_summary(order: Order) -> dict:
    """Get a summary of the order for display."""
    return {
        "order_id": order.id,
        "items": len(order.items),
        "subtotal": f"${order.subtotal:.2f}",
        "discount": f"-${order.discount_amount:.2f}" if order.discount_amount > 0 else "$0.00",
        "tax": f"${order.tax_amount:.2f}",
        "total": f"${order.total:.2f}",
        "status": order.status.value,
    }
