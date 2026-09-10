"""API endpoints that use event logging."""

from config_loader import get_config
from event_logger import EventLogger, log_event


class UserAPI:
    """User management endpoints."""

    def __init__(self):
        self.logger = EventLogger(source="user_api")
        self.config = get_config()

    def create_user(self, username: str, email: str) -> dict:
        """Create a new user."""
        # Log the event
        event = self.logger.log("user_created", {
            "username": username,
            "email": email,
        })

        return {
            "success": True,
            "username": username,
            "created_at": event.timestamp,
        }

    def login(self, username: str) -> dict:
        """Log in a user."""
        event = self.logger.log("user_login", {"username": username})

        return {
            "success": True,
            "username": username,
            "logged_in_at": event.timestamp,
        }


class OrderAPI:
    """Order management endpoints - uses default logger."""

    def __init__(self):
        self.config = get_config()

    def create_order(self, user_id: str, items: list) -> dict:
        """Create a new order."""
        event = log_event("order_created", {
            "user_id": user_id,
            "item_count": len(items),
        })

        return {
            "success": True,
            "order_id": "ord_12345",
            "created_at": event.timestamp,
        }
