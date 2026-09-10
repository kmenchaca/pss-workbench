"""Safety rails for the Adversarial Red Team Engine.

Provides guardrails to ensure red team testing doesn't cause actual harm:
sandboxing, rate limiting, scope restrictions, and audit logging.
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .types import AttackVector, TargetSystem


@dataclass
class AuditEntry:
    """A single audit log entry.

    Attributes:
        timestamp: When the action occurred.
        action: Type of action (probe, spawn, verify, etc.).
        target: Target being tested.
        payload_hash: SHA256 hash of payload (not storing raw payload in logs).
        payload_preview: First 100 chars of payload for reference.
        result: Result of the action.
        blocked: Whether the action was blocked.
        reason: Reason for blocking (if blocked).
        session_id: Session this action belongs to.
    """
    timestamp: datetime
    action: str
    target: str
    payload_hash: str
    payload_preview: str
    result: str
    blocked: bool = False
    reason: str = ""
    session_id: str = ""


@dataclass
class ScopeConfig:
    """Configuration for authorized testing scope.

    Attributes:
        allowed_targets: List of allowed target identifiers.
        allowed_categories: List of allowed attack categories.
        blocked_patterns: Patterns that should never be sent.
        max_payload_length: Maximum payload length in characters.
        require_confirmation: Whether to require confirmation for high-severity.
    """
    allowed_targets: list[str] = field(default_factory=list)
    allowed_categories: list[str] = field(default_factory=list)
    blocked_patterns: list[str] = field(default_factory=list)
    max_payload_length: int = 10000
    require_confirmation: bool = False


@dataclass
class RateLimitConfig:
    """Rate limiting configuration.

    Attributes:
        max_probes_per_minute: Maximum probes per minute.
        max_probes_per_target: Maximum probes per target per session.
        cooldown_seconds: Cooldown after hitting limit.
        burst_limit: Maximum burst of rapid probes.
    """
    max_probes_per_minute: int = 60
    max_probes_per_target: int = 1000
    cooldown_seconds: int = 60
    burst_limit: int = 10


@dataclass
class SandboxConfig:
    """Sandboxing configuration.

    Attributes:
        enabled: Whether sandboxing is enabled.
        dry_run: If true, don't actually send probes.
        capture_only: Only capture what would be sent.
        allow_network: Whether to allow network access.
        timeout_seconds: Maximum time for any operation.
    """
    enabled: bool = True
    dry_run: bool = False
    capture_only: bool = False
    allow_network: bool = True
    timeout_seconds: int = 30


class SafetyGuard:
    """Safety guardrails for red team testing.

    Prevents actual harm by enforcing:
    - Scope restrictions (only test authorized targets)
    - Rate limiting (prevent DoS)
    - Content filtering (block certain payloads)
    - Audit logging (track all attempts)
    - Sandboxing (isolate tests)
    """

    def __init__(
        self,
        scope: ScopeConfig | None = None,
        rate_limit: RateLimitConfig | None = None,
        sandbox: SandboxConfig | None = None,
        audit_log_path: Path | None = None,
    ) -> None:
        """Initialize safety guard.

        Args:
            scope: Scope configuration.
            rate_limit: Rate limit configuration.
            sandbox: Sandbox configuration.
            audit_log_path: Path to write audit logs.
        """
        self.scope = scope or ScopeConfig()
        self.rate_limit = rate_limit or RateLimitConfig()
        self.sandbox = sandbox or SandboxConfig()
        self.audit_log_path = audit_log_path

        # Rate limiting state
        self._probe_timestamps: list[float] = []
        self._probes_per_target: dict[str, int] = {}
        self._cooldown_until: float = 0

        # Audit log
        self._audit_log: list[AuditEntry] = []
        self._session_id = self._generate_session_id()

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return hashlib.sha256(
            f"{time.time()}-{id(self)}".encode()
        ).hexdigest()[:12]

    def check_scope(self, target: TargetSystem, payload: str) -> tuple[bool, str]:
        """Check if an attack is within authorized scope.

        Args:
            target: Target to attack.
            payload: Payload to send.

        Returns:
            Tuple of (allowed, reason).
        """
        # Check target is allowed
        if self.scope.allowed_targets:
            if target.name not in self.scope.allowed_targets:
                return False, f"Target '{target.name}' not in allowed targets"

        # Check payload length
        if len(payload) > self.scope.max_payload_length:
            return False, f"Payload exceeds max length ({self.scope.max_payload_length})"

        # Check blocked patterns
        for pattern in self.scope.blocked_patterns:
            if re.search(pattern, payload, re.IGNORECASE):
                return False, f"Payload matches blocked pattern: {pattern}"

        return True, ""

    def check_rate_limit(self, target: str) -> tuple[bool, str]:
        """Check if rate limit allows another probe.

        Args:
            target: Target identifier.

        Returns:
            Tuple of (allowed, reason).
        """
        current_time = time.time()

        # Check cooldown
        if current_time < self._cooldown_until:
            remaining = int(self._cooldown_until - current_time)
            return False, f"In cooldown, {remaining}s remaining"

        # Clean old timestamps
        cutoff = current_time - 60
        self._probe_timestamps = [
            t for t in self._probe_timestamps if t > cutoff
        ]

        # Check probes per minute
        if len(self._probe_timestamps) >= self.rate_limit.max_probes_per_minute:
            self._cooldown_until = current_time + self.rate_limit.cooldown_seconds
            return False, f"Rate limit exceeded ({self.rate_limit.max_probes_per_minute}/min)"

        # Check burst limit
        recent_cutoff = current_time - 1  # Last second
        recent_probes = sum(1 for t in self._probe_timestamps if t > recent_cutoff)
        if recent_probes >= self.rate_limit.burst_limit:
            return False, f"Burst limit exceeded ({self.rate_limit.burst_limit}/sec)"

        # Check per-target limit
        target_probes = self._probes_per_target.get(target, 0)
        if target_probes >= self.rate_limit.max_probes_per_target:
            return False, f"Target limit exceeded ({self.rate_limit.max_probes_per_target})"

        return True, ""

    def record_probe(self, target: str) -> None:
        """Record a probe for rate limiting.

        Args:
            target: Target identifier.
        """
        self._probe_timestamps.append(time.time())
        self._probes_per_target[target] = self._probes_per_target.get(target, 0) + 1

    def validate_attack(
        self,
        vector: AttackVector,
        target: TargetSystem,
    ) -> tuple[bool, str]:
        """Validate an attack vector before execution.

        Args:
            vector: Attack vector to validate.
            target: Target to attack.

        Returns:
            Tuple of (allowed, reason).
        """
        # Check scope
        allowed, reason = self.check_scope(target, vector.payload)
        if not allowed:
            self._log_audit(
                action="validate",
                target=target.name,
                payload=vector.payload,
                result="blocked",
                blocked=True,
                reason=reason,
            )
            return False, reason

        # Check category is allowed
        if self.scope.allowed_categories:
            if vector.category.value not in self.scope.allowed_categories:
                reason = f"Category '{vector.category.value}' not allowed"
                self._log_audit(
                    action="validate",
                    target=target.name,
                    payload=vector.payload,
                    result="blocked",
                    blocked=True,
                    reason=reason,
                )
                return False, reason

        # Check rate limit
        allowed, reason = self.check_rate_limit(target.name)
        if not allowed:
            self._log_audit(
                action="validate",
                target=target.name,
                payload=vector.payload,
                result="rate_limited",
                blocked=True,
                reason=reason,
            )
            return False, reason

        return True, ""

    def pre_probe(
        self,
        target: TargetSystem,
        payload: str,
    ) -> tuple[bool, str | None]:
        """Pre-probe safety check.

        Args:
            target: Target to probe.
            payload: Payload to send.

        Returns:
            Tuple of (proceed, modified_payload).
            If proceed is False, don't send the probe.
            If modified_payload is not None, use it instead.
        """
        # Check sandbox mode
        if self.sandbox.dry_run:
            self._log_audit(
                action="probe_dry_run",
                target=target.name,
                payload=payload,
                result="captured",
            )
            return False, None

        if self.sandbox.capture_only:
            self._log_audit(
                action="probe_capture",
                target=target.name,
                payload=payload,
                result="captured",
            )
            return False, None

        # Validate
        allowed, reason = self.check_scope(target, payload)
        if not allowed:
            self._log_audit(
                action="probe",
                target=target.name,
                payload=payload,
                result="blocked",
                blocked=True,
                reason=reason,
            )
            return False, None

        # Check rate limit
        allowed, reason = self.check_rate_limit(target.name)
        if not allowed:
            self._log_audit(
                action="probe",
                target=target.name,
                payload=payload,
                result="rate_limited",
                blocked=True,
                reason=reason,
            )
            return False, None

        # Record the probe
        self.record_probe(target.name)

        return True, None

    def post_probe(
        self,
        target: TargetSystem,
        payload: str,
        response: str,
        success: bool,
    ) -> None:
        """Post-probe logging.

        Args:
            target: Target that was probed.
            payload: Payload that was sent.
            response: Response received.
            success: Whether attack was successful.
        """
        result = "success" if success else "no_vuln"
        self._log_audit(
            action="probe_complete",
            target=target.name,
            payload=payload,
            result=result,
        )

    def _log_audit(
        self,
        action: str,
        target: str,
        payload: str,
        result: str,
        blocked: bool = False,
        reason: str = "",
    ) -> None:
        """Add an entry to the audit log.

        Args:
            action: Action type.
            target: Target identifier.
            payload: Payload (will be hashed).
            result: Result of action.
            blocked: Whether action was blocked.
            reason: Reason for blocking.
        """
        entry = AuditEntry(
            timestamp=datetime.now(),
            action=action,
            target=target,
            payload_hash=hashlib.sha256(payload.encode()).hexdigest(),
            payload_preview=payload[:100] + ("..." if len(payload) > 100 else ""),
            result=result,
            blocked=blocked,
            reason=reason,
            session_id=self._session_id,
        )
        self._audit_log.append(entry)

        # Write to file if configured
        if self.audit_log_path:
            self._write_audit_entry(entry)

    def _write_audit_entry(self, entry: AuditEntry) -> None:
        """Write audit entry to file.

        Args:
            entry: Entry to write.
        """
        if not self.audit_log_path:
            return

        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.audit_log_path, "a") as f:
            f.write(json.dumps({
                "timestamp": entry.timestamp.isoformat(),
                "action": entry.action,
                "target": entry.target,
                "payload_hash": entry.payload_hash,
                "payload_preview": entry.payload_preview,
                "result": entry.result,
                "blocked": entry.blocked,
                "reason": entry.reason,
                "session_id": entry.session_id,
            }) + "\n")

    def get_audit_log(self) -> list[AuditEntry]:
        """Get the full audit log.

        Returns:
            List of audit entries.
        """
        return self._audit_log.copy()

    def get_session_stats(self) -> dict[str, Any]:
        """Get statistics for the current session.

        Returns:
            Dictionary of session statistics.
        """
        total = len(self._audit_log)
        blocked = sum(1 for e in self._audit_log if e.blocked)
        successful = sum(1 for e in self._audit_log if e.result == "success")

        return {
            "session_id": self._session_id,
            "total_actions": total,
            "blocked_actions": blocked,
            "successful_probes": successful,
            "probes_per_target": dict(self._probes_per_target),
            "in_cooldown": time.time() < self._cooldown_until,
        }

    def reset_rate_limits(self) -> None:
        """Reset rate limiting state."""
        self._probe_timestamps = []
        self._probes_per_target = {}
        self._cooldown_until = 0


def create_permissive_guard() -> SafetyGuard:
    """Create a permissive safety guard for testing.

    Returns:
        SafetyGuard with minimal restrictions.
    """
    return SafetyGuard(
        scope=ScopeConfig(max_payload_length=100000),
        rate_limit=RateLimitConfig(
            max_probes_per_minute=1000,
            max_probes_per_target=10000,
        ),
        sandbox=SandboxConfig(enabled=False),
    )


def create_strict_guard(
    allowed_targets: list[str],
    allowed_categories: list[str] | None = None,
    audit_log_path: Path | None = None,
) -> SafetyGuard:
    """Create a strict safety guard for production use.

    Args:
        allowed_targets: Only these targets can be tested.
        allowed_categories: Only these categories allowed.
        audit_log_path: Path for audit logs.

    Returns:
        SafetyGuard with strict restrictions.
    """
    return SafetyGuard(
        scope=ScopeConfig(
            allowed_targets=allowed_targets,
            allowed_categories=allowed_categories or [],
            max_payload_length=5000,
            require_confirmation=True,
        ),
        rate_limit=RateLimitConfig(
            max_probes_per_minute=30,
            max_probes_per_target=100,
            cooldown_seconds=120,
            burst_limit=5,
        ),
        sandbox=SandboxConfig(
            enabled=True,
            timeout_seconds=10,
        ),
        audit_log_path=audit_log_path,
    )


def create_dry_run_guard() -> SafetyGuard:
    """Create a dry-run guard that captures but doesn't send.

    Returns:
        SafetyGuard in dry-run mode.
    """
    return SafetyGuard(
        sandbox=SandboxConfig(
            enabled=True,
            dry_run=True,
        ),
    )
