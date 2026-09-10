"""Tests for order processing functionality."""

import pytest
from ..models import OrderStatus
from ..orders import create_order, add_item_to_order, finalize_order, get_order_summary
from ..catalog import get_product, get_all_products, load_catalog, clear_catalog
from ..models import Product


class TestOrderCreation:
    """Tests for creating orders."""

    def test_create_order_basic(self, catalog_with_products):
        """Test creating a basic order."""
        order = create_order("customer-1")

        assert order.customer_id == "customer-1"
        assert order.status == OrderStatus.PENDING
        assert len(order.items) == 0

    def test_add_item_to_order(self, catalog_with_products):
        """Test adding items to an order."""
        order = create_order("customer-2")

        result = add_item_to_order(order, "LAPTOP")

        assert result is None  # No error
        assert len(order.items) == 1
        assert order.items[0].product_id == "LAPTOP"
        assert order.items[0].unit_price == 999.99


class TestOrderCalculations:
    """Tests for order total calculations."""

    def test_order_subtotal(self, catalog_with_products):
        """Test that subtotal is calculated correctly."""
        order = create_order("customer-3")
        add_item_to_order(order, "LAPTOP")  # $999.99
        add_item_to_order(order, "MOUSE")   # $29.99

        assert order.subtotal == pytest.approx(1029.98)

    def test_order_with_quantity(self, catalog_with_products):
        """Test order with multiple quantities."""
        order = create_order("customer-4")
        add_item_to_order(order, "BOOK", quantity=3)  # $49.99 * 3

        assert order.subtotal == pytest.approx(149.97)

    def test_tax_calculation_california(self, catalog_with_products):
        """Test tax calculation for California."""
        order = create_order("customer-5", state="CA")
        add_item_to_order(order, "MOUSE")  # $29.99

        # CA tax is 7.25%
        expected_tax = 29.99 * 0.0725
        assert order.tax_amount == pytest.approx(expected_tax, rel=0.01)


class TestOrderDiscount:
    """Tests for order discounts."""

    def test_tier_discount_applied(self, catalog_with_products):
        """Test that tier discount is applied on finalize."""
        order = create_order("customer-6")
        add_item_to_order(order, "LAPTOP")  # $999.99 - should get 15% off

        finalize_order(order)

        assert order.discount_percent == 15.0
        assert order.status == OrderStatus.CONFIRMED


class TestCatalogIsolation:
    """Tests that verify catalog isolation between tests.

    IMPORTANT: These tests verify that each test gets a fresh catalog.
    If these fail, it indicates test pollution.
    """

    def test_catalog_has_exactly_five_products(self, catalog_with_products):
        """
        CRITICAL: Catalog should have exactly 5 test products.

        If this test fails with more products, there's test pollution.
        """
        products = get_all_products()

        assert len(products) == 5, \
            f"Expected 5 products, got {len(products)}: {[p.id for p in products]}"

    def test_add_custom_product_isolated(self, catalog_with_products):
        """Test that adding a product doesn't leak to other tests."""
        # Add a custom product for this test
        custom_product = Product(
            id="CUSTOM",
            name="Custom Test Product",
            price=99.99,
            category="test"
        )
        load_catalog([custom_product])

        # Now we should have 6 products
        products = get_all_products()
        assert len(products) == 6

    def test_catalog_still_has_five_products(self, catalog_with_products):
        """
        CRITICAL: After previous test, catalog should still have only 5 products.

        The previous test added a product. If this test sees 6 products,
        it means the fixture isn't properly isolating test data.

        This is the test that FAILS due to the bug.
        """
        products = get_all_products()

        assert len(products) == 5, \
            f"TEST POLLUTION: Expected 5 products but got {len(products)}. " \
            f"Products: {[p.id for p in products]}"


class TestOrderFinalization:
    """Tests for finalizing orders."""

    def test_finalize_order_success(self, catalog_with_products):
        """Test successful order finalization."""
        order = create_order("customer-7")
        add_item_to_order(order, "KEYBOARD")

        result = finalize_order(order)

        assert result is None
        assert order.status == OrderStatus.CONFIRMED

    def test_finalize_empty_order_fails(self, catalog_with_products):
        """Test that finalizing empty order fails."""
        order = create_order("customer-8")

        result = finalize_order(order)

        assert result == "Cannot finalize empty order"
        assert order.status == OrderStatus.PENDING
