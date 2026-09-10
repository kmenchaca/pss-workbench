"""Data models for order processing."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


@dataclass
class Product:
    """Product in the catalog."""
    id: str
    name: str
    price: float
    category: str
    in_stock: bool = True


@dataclass
class OrderItem:
    """An item in an order."""
    product_id: str
    quantity: int
    unit_price: float

    @property
    def subtotal(self) -> float:
        return self.quantity * self.unit_price


@dataclass
class Order:
    """A customer order."""
    id: str
    customer_id: str
    items: list[OrderItem] = field(default_factory=list)
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    discount_percent: float = 0.0
    tax_rate: float = 0.08  # 8% tax

    @property
    def subtotal(self) -> float:
        """Sum of all item subtotals."""
        return sum(item.subtotal for item in self.items)

    @property
    def discount_amount(self) -> float:
        """Calculate discount."""
        return self.subtotal * (self.discount_percent / 100)

    @property
    def tax_amount(self) -> float:
        """Calculate tax on discounted amount."""
        return (self.subtotal - self.discount_amount) * self.tax_rate

    @property
    def total(self) -> float:
        """Final order total."""
        return self.subtotal - self.discount_amount + self.tax_amount
