"""Event logging system with timestamps."""

from dataclasses import dataclass
from typing import Any
from time_utils import get_current_time, format_timestamp


@dataclass
class Event:
    event_type: str
    data: dict
    timestamp: str
    source: str


class EventLogger:
    """Logs events with timestamps."""

    def __init__(self, source: str = "unknown"):
        self.source = source
        self._events: list[Event] = []

    def log(self, event_type: str, data: dict = None) -> Event:
        """Log an event with the current timestamp."""
        event = Event(
            event_type=event_type,
            data=data or {},
            timestamp=format_timestamp(get_current_time()),
            source=self.source,
        )
        self._events.append(event)
        return event

    def get_events(self) -> list[Event]:
        """Get all logged events."""
        return self._events.copy()


# Convenience function for quick logging
_default_logger = EventLogger(source="default")


def log_event(event_type: str, data: dict = None) -> Event:
    """Log an event using the default logger."""
    return _default_logger.log(event_type, data)
