"""Buggy application for PSS debugging demo.

THE BUG: Large file uploads fail with "PayloadSerializer OOM" error.
Stack trace points at the serializer. But the serializer is fine.

REAL CAUSE: The `@log_request_body` decorator buffers the entire request
body into memory BEFORE the serializer even runs. For large files, this
causes OOM, but the error surfaces later in the serializer because that's
where the memory allocation finally fails.

This is a realistic misdirection - the stack trace is technically accurate
(the OOM happens during serialization) but the ROOT CAUSE is the decorator.
"""

import io
import logging
from dataclasses import dataclass
from functools import wraps
from typing import Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# THE HIDDEN BUG: This decorator buffers entire request body
# ============================================================================

def log_request_body(func: Callable) -> Callable:
    """Decorator that logs request bodies for debugging.

    BUG: This reads the ENTIRE body into memory before passing to handler.
    For large file uploads, this causes OOM before serialization even starts.
    The OOM surfaces later in the serializer, creating a misleading stack trace.
    """
    @wraps(func)
    def wrapper(request: "Request", *args, **kwargs):
        # BUG: Buffering entire body into memory
        body_copy = request.body.read()  # <-- THIS IS THE BUG
        request.body = io.BytesIO(body_copy)  # Reset for handler

        # Log truncated version (seems innocent)
        logger.debug(f"Request body preview: {body_copy[:100]}...")

        return func(request, *args, **kwargs)
    return wrapper


# ============================================================================
# THE INNOCENT SERIALIZER (blamed by stack trace)
# ============================================================================

class PayloadSerializer:
    """Serializes upload payloads. Blamed by stack trace but actually fine."""

    def __init__(self, chunk_size: int = 8192):
        self.chunk_size = chunk_size

    def serialize(self, data: bytes) -> dict:
        """Serialize payload data.

        This method is BLAMED in the stack trace, but it's actually fine.
        It processes data in chunks and doesn't buffer everything.

        The OOM happens here because by the time we get here, the decorator
        has already exhausted memory. This allocation is just the straw
        that breaks the camel's back.
        """
        # This allocation triggers the OOM, but isn't the root cause
        result = {
            "size": len(data),
            "checksum": hash(data) % (10**9),
            "chunks": []
        }

        # Process in chunks (this is actually efficient)
        for i in range(0, len(data), self.chunk_size):
            chunk = data[i:i + self.chunk_size]
            result["chunks"].append({
                "offset": i,
                "size": len(chunk),
                "preview": chunk[:20].hex() if chunk else ""
            })

        return result


# ============================================================================
# REQUEST HANDLING
# ============================================================================

@dataclass
class Request:
    """Simulated HTTP request."""
    body: io.BytesIO
    content_length: int
    content_type: str = "application/octet-stream"


class RequestHandler:
    """Handles file upload requests."""

    def __init__(self):
        self.serializer = PayloadSerializer()

    @log_request_body  # <-- THE BUG IS HERE (decorator)
    def process_upload(self, request: Request) -> dict:
        """Process an uploaded file.

        Stack trace will show this calling serializer.serialize(),
        but the real issue is the decorator above this method.
        """
        logger.info(f"Processing upload: {request.content_length} bytes")

        # Read body (already buffered by decorator, so this is "free")
        data = request.body.read()

        # Serialize - THIS IS WHERE OOM SURFACES (but not the cause)
        try:
            result = self.serializer.serialize(data)
            return {"status": "success", "payload": result}
        except MemoryError as e:
            # This is what the user sees in logs
            logger.error(f"PayloadSerializer.serialize() failed: {e}")
            raise


class UploadController:
    """HTTP controller for uploads."""

    def __init__(self):
        self.handler = RequestHandler()

    def handle_file_upload(self, request: Request) -> dict:
        """Entry point for file uploads."""
        return self.handler.process_upload(request)


# ============================================================================
# SIMULATED ERROR REPRODUCTION
# ============================================================================

def simulate_large_upload(size_mb: int = 100):
    """Simulate a large file upload that triggers the bug.

    In real life, this would be triggered by actual HTTP request.
    The error would show:

    ERROR PayloadSerializer.serialize() failed
      at PayloadSerializer.serialize(buggy_app.py:62)
      at RequestHandler.process_upload(buggy_app.py:91)
      at UploadController.handle_file_upload(buggy_app.py:103)
    MemoryError: Unable to allocate array

    The stack trace points at the serializer, but the decorator is the cause.
    """
    # Create large payload
    data = b"x" * (size_mb * 1024 * 1024)
    request = Request(
        body=io.BytesIO(data),
        content_length=len(data),
    )

    controller = UploadController()
    return controller.handle_file_upload(request)


# ============================================================================
# THE FIX (for verification)
# ============================================================================

def log_request_body_fixed(func: Callable) -> Callable:
    """Fixed version: only log content-length, don't buffer body."""
    @wraps(func)
    def wrapper(request: "Request", *args, **kwargs):
        # FIXED: Just log metadata, don't read body
        logger.debug(f"Request content-length: {request.content_length}")
        return func(request, *args, **kwargs)
    return wrapper


# ============================================================================
# GROUND TRUTH
# ============================================================================

GROUND_TRUTH = {
    "bug_location": "log_request_body decorator in buggy_app.py",
    "bug_line": "body_copy = request.body.read()",
    "why_misleading": "Stack trace shows OOM in serializer, but decorator buffers entire body first",
    "fix": "Remove body buffering from decorator, or use streaming",
    "red_herrings": [
        "PayloadSerializer.serialize()",
        "Increase heap size",
        "Process in smaller chunks",
        "Use streaming serialization",
    ],
    "correct_insights": [
        "decorator",
        "log_request_body",
        "buffers",
        "before serialization",
        "middleware",
        "request handling",
        "not the serializer",
    ]
}


if __name__ == "__main__":
    print("Buggy App - File Upload Handler")
    print("=" * 50)
    print()
    print("THE BUG:")
    print("  Large file uploads fail with 'PayloadSerializer OOM'")
    print("  Stack trace points at serializer.serialize()")
    print()
    print("THE TRUTH:")
    print(f"  Bug location: {GROUND_TRUTH['bug_location']}")
    print(f"  Bug line: {GROUND_TRUTH['bug_line']}")
    print()
    print("RED HERRINGS (what single-shot will suggest):")
    for herring in GROUND_TRUTH['red_herrings']:
        print(f"  - {herring}")
    print()
    print("CORRECT INSIGHTS (what we're looking for):")
    for insight in GROUND_TRUTH['correct_insights']:
        print(f"  - {insight}")
