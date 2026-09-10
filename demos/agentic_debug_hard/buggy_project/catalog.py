"""Product catalog management."""

from typing import Optional
from .models import Product


# Module-level catalog storage
_catalog: dict[str, Product] = {}


def load_catalog(products: list[Product]) -> None:
    """Load products into the catalog."""
    global _catalog
    for product in products:
        _catalog[product.id] = product


def get_product(product_id: str) -> Optional[Product]:
    """Get a product by ID."""
    return _catalog.get(product_id)


def get_all_products() -> list[Product]:
    """Get all products in the catalog."""
    return list(_catalog.values())


def clear_catalog() -> None:
    """Clear the catalog (for testing)."""
    global _catalog
    _catalog = {}


def get_products_by_category(category: str) -> list[Product]:
    """Get all products in a category."""
    return [p for p in _catalog.values() if p.category == category]
