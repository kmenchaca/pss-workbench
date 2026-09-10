"""Pytest configuration and fixtures for order processing tests.

This file provides shared test fixtures used across all test modules.
"""

import pytest
from ..models import Product
from ..catalog import load_catalog, clear_catalog


# Standard test products
TEST_PRODUCTS = [
    Product(id="LAPTOP", name="Laptop Pro", price=999.99, category="electronics"),
    Product(id="MOUSE", name="Wireless Mouse", price=29.99, category="electronics"),
    Product(id="KEYBOARD", name="Mechanical Keyboard", price=149.99, category="electronics"),
    Product(id="HEADPHONES", name="Noise Canceling Headphones", price=199.99, category="electronics"),
    Product(id="BOOK", name="Python Programming", price=49.99, category="books"),
]


@pytest.fixture(scope="module")  # BUG: Should be scope="function"
def catalog_with_products():
    """
    Load test products into the catalog.

    NOTE: Using module scope for performance - catalog is expensive to load.
    Products are cleared after the module completes.
    """
    clear_catalog()
    load_catalog(TEST_PRODUCTS)
    yield TEST_PRODUCTS
    clear_catalog()


@pytest.fixture
def empty_catalog():
    """Provide an empty catalog for tests that need it."""
    clear_catalog()
    yield
    clear_catalog()


@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    from ..orders import create_order, add_item_to_order

    order = create_order("customer-123", state="CA")
    # Note: Requires catalog_with_products fixture to work
    return order
