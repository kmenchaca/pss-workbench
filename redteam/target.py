"""Target abstraction layer for the Adversarial Red Team Engine.

Provides base classes and implementations for different attack targets:
LLM models, HTTP APIs, and prompt-specific targets.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from .types import TargetSystem, TargetType


@dataclass
class ProbeResult:
    """Result of probing a target.

    Attributes:
        payload: The payload that was sent.
        response: Raw response from the target.
        status_code: HTTP status code (for API targets).
        duration_ms: How long the probe took.
        error: Error message if probe failed.
        metadata: Additional target-specific data.
    """
    payload: str
    response: str
    status_code: int | None = None
    duration_ms: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Target(ABC):
    """Base class for attack targets.

    All target implementations must provide a probe method that sends
    a payload and returns the target's response.
    """

    def __init__(self, system: TargetSystem) -> None:
        """Initialize target with system configuration.

        Args:
            system: Target system configuration.
        """
        self.system = system
        self._probe_count = 0
        self._last_probe_time: float | None = None

    @property
    def name(self) -> str:
        """Get target name."""
        return self.system.name

    @property
    def endpoint(self) -> str:
        """Get target endpoint."""
        return self.system.endpoint

    @abstractmethod
    async def probe(self, payload: str) -> ProbeResult:
        """Send a payload to the target and get response.

        Args:
            payload: The attack payload to send.

        Returns:
            ProbeResult containing response and metadata.
        """
        pass

    def is_in_scope(self, payload: str) -> bool:
        """Check if a payload is within authorized scope.

        Args:
            payload: Payload to check.

        Returns:
            True if payload is in scope.
        """
        if not self.system.scope:
            return True
        # Check if payload matches any scope pattern
        return any(scope in payload for scope in self.system.scope)

    def detect_data_leakage(self, response: str) -> list[str]:
        """Detect potential data leakage in response.

        Args:
            response: Response to analyze.

        Returns:
            List of matched sensitive patterns.
        """
        matches = []
        for pattern in self.system.sensitive_patterns:
            if pattern.lower() in response.lower():
                matches.append(pattern)
        return matches

    def detect_policy_violation(self, response: str, policy_rules: list[str]) -> list[str]:
        """Detect policy violations in response.

        Args:
            response: Response to analyze.
            policy_rules: List of policy rules (patterns that shouldn't appear).

        Returns:
            List of violated rules.
        """
        violations = []
        for rule in policy_rules:
            if rule.lower() in response.lower():
                violations.append(rule)
        return violations

    def detect_unexpected_behavior(
        self,
        baseline_response: str,
        current_response: str,
        similarity_threshold: float = 0.5
    ) -> bool:
        """Detect unexpected behavior by comparing to baseline.

        Args:
            baseline_response: Expected baseline response.
            current_response: Current response to compare.
            similarity_threshold: Threshold below which behavior is unexpected.

        Returns:
            True if behavior is unexpected.
        """
        # Simple word overlap similarity
        baseline_words = set(baseline_response.lower().split())
        current_words = set(current_response.lower().split())

        if not baseline_words or not current_words:
            return True

        overlap = len(baseline_words & current_words)
        total = len(baseline_words | current_words)

        similarity = overlap / total if total > 0 else 0
        return similarity < similarity_threshold


class LLMTarget(Target):
    """Target wrapper for LLM models.

    Wraps any LLM as an attack target, supporting both direct SDK calls
    and custom call functions.
    """

    def __init__(
        self,
        system: TargetSystem,
        call_fn: Callable[[str], str] | None = None,
        async_call_fn: Callable[[str], Any] | None = None,
    ) -> None:
        """Initialize LLM target.

        Args:
            system: Target system configuration.
            call_fn: Synchronous function to call the LLM.
            async_call_fn: Async function to call the LLM.
        """
        super().__init__(system)
        self._call_fn = call_fn
        self._async_call_fn = async_call_fn

        if not call_fn and not async_call_fn:
            raise ValueError("Must provide either call_fn or async_call_fn")

    async def probe(self, payload: str) -> ProbeResult:
        """Send payload to LLM and get response.

        Args:
            payload: The attack payload to send.

        Returns:
            ProbeResult with LLM response.
        """
        start_time = time.monotonic()
        self._probe_count += 1

        try:
            if self._async_call_fn:
                response = await self._async_call_fn(payload)
            elif self._call_fn:
                response = await asyncio.to_thread(self._call_fn, payload)
            else:
                raise RuntimeError("No call function configured")

            duration_ms = int((time.monotonic() - start_time) * 1000)
            self._last_probe_time = time.monotonic()

            return ProbeResult(
                payload=payload,
                response=str(response),
                duration_ms=duration_ms,
                metadata={
                    "model": self.system.endpoint,
                    "probe_count": self._probe_count,
                }
            )
        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return ProbeResult(
                payload=payload,
                response="",
                duration_ms=duration_ms,
                error=str(e),
            )


class APITarget(Target):
    """Target wrapper for HTTP APIs.

    Wraps any HTTP API endpoint as an attack target.
    """

    def __init__(
        self,
        system: TargetSystem,
        method: str = "POST",
        headers: dict[str, str] | None = None,
        payload_template: str | None = None,
        response_extractor: Callable[[httpx.Response], str] | None = None,
    ) -> None:
        """Initialize API target.

        Args:
            system: Target system configuration.
            method: HTTP method to use.
            headers: Request headers.
            payload_template: Template for request body (use {payload} placeholder).
            response_extractor: Function to extract response text.
        """
        super().__init__(system)
        self._method = method
        self._headers = headers or {}
        self._payload_template = payload_template or '{"input": "{payload}"}'
        self._response_extractor = response_extractor

    async def probe(self, payload: str) -> ProbeResult:
        """Send payload to API and get response.

        Args:
            payload: The attack payload to send.

        Returns:
            ProbeResult with API response.
        """
        start_time = time.monotonic()
        self._probe_count += 1

        # Escape payload for JSON
        escaped_payload = payload.replace('"', '\\"').replace('\n', '\\n')
        body = self._payload_template.replace("{payload}", escaped_payload)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.request(
                    method=self._method,
                    url=self.system.endpoint,
                    content=body,
                    headers={
                        "Content-Type": "application/json",
                        **self._headers,
                    },
                )

                duration_ms = int((time.monotonic() - start_time) * 1000)
                self._last_probe_time = time.monotonic()

                if self._response_extractor:
                    response_text = self._response_extractor(response)
                else:
                    response_text = response.text

                return ProbeResult(
                    payload=payload,
                    response=response_text,
                    status_code=response.status_code,
                    duration_ms=duration_ms,
                    metadata={
                        "endpoint": self.system.endpoint,
                        "method": self._method,
                        "probe_count": self._probe_count,
                    }
                )
        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return ProbeResult(
                payload=payload,
                response="",
                duration_ms=duration_ms,
                error=str(e),
            )


class PromptTarget(Target):
    """Target specifically for prompt injection testing.

    Wraps an LLM with a system prompt, testing if the system prompt
    can be bypassed or leaked.
    """

    def __init__(
        self,
        system: TargetSystem,
        system_prompt: str,
        call_fn: Callable[[str, str], str] | None = None,
        async_call_fn: Callable[[str, str], Any] | None = None,
    ) -> None:
        """Initialize prompt target.

        Args:
            system: Target system configuration.
            system_prompt: The system prompt to test against.
            call_fn: Sync function taking (system_prompt, user_input).
            async_call_fn: Async function taking (system_prompt, user_input).
        """
        super().__init__(system)
        self._system_prompt = system_prompt
        self._call_fn = call_fn
        self._async_call_fn = async_call_fn

        if not call_fn and not async_call_fn:
            raise ValueError("Must provide either call_fn or async_call_fn")

    @property
    def system_prompt(self) -> str:
        """Get the system prompt being tested."""
        return self._system_prompt

    async def probe(self, payload: str) -> ProbeResult:
        """Send payload as user input, testing against system prompt.

        Args:
            payload: The attack payload (user input).

        Returns:
            ProbeResult with response.
        """
        start_time = time.monotonic()
        self._probe_count += 1

        try:
            if self._async_call_fn:
                response = await self._async_call_fn(self._system_prompt, payload)
            elif self._call_fn:
                response = await asyncio.to_thread(
                    self._call_fn, self._system_prompt, payload
                )
            else:
                raise RuntimeError("No call function configured")

            duration_ms = int((time.monotonic() - start_time) * 1000)
            self._last_probe_time = time.monotonic()

            return ProbeResult(
                payload=payload,
                response=str(response),
                duration_ms=duration_ms,
                metadata={
                    "model": self.system.endpoint,
                    "has_system_prompt": True,
                    "probe_count": self._probe_count,
                }
            )
        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return ProbeResult(
                payload=payload,
                response="",
                duration_ms=duration_ms,
                error=str(e),
            )

    def detect_system_prompt_leak(self, response: str) -> tuple[bool, list[str]]:
        """Detect if system prompt was leaked in response.

        Args:
            response: Response to analyze.

        Returns:
            Tuple of (leaked, evidence) where evidence is matching fragments.
        """
        evidence = []
        response_lower = response.lower()
        prompt_lower = self._system_prompt.lower()

        # Check for direct inclusion of system prompt fragments
        words = prompt_lower.split()
        for i in range(len(words) - 2):
            fragment = " ".join(words[i:i+3])
            if fragment in response_lower and len(fragment) > 10:
                evidence.append(fragment)

        # Check for full inclusion
        if prompt_lower in response_lower:
            evidence.append("[FULL SYSTEM PROMPT LEAKED]")

        return bool(evidence), evidence


def create_llm_target(
    name: str,
    endpoint: str,
    call_fn: Callable[[str], str],
    sensitive_patterns: list[str] | None = None,
) -> LLMTarget:
    """Factory function to create an LLM target.

    Args:
        name: Human-readable name.
        endpoint: Model identifier or URL.
        call_fn: Function to call the model.
        sensitive_patterns: Patterns indicating sensitive data.

    Returns:
        Configured LLMTarget.
    """
    system = TargetSystem(
        type=TargetType.LLM,
        name=name,
        endpoint=endpoint,
        interface="sdk",
        sensitive_patterns=sensitive_patterns or [],
    )
    return LLMTarget(system, call_fn=call_fn)


def create_api_target(
    name: str,
    endpoint: str,
    method: str = "POST",
    headers: dict[str, str] | None = None,
    sensitive_patterns: list[str] | None = None,
) -> APITarget:
    """Factory function to create an API target.

    Args:
        name: Human-readable name.
        endpoint: API URL.
        method: HTTP method.
        headers: Request headers.
        sensitive_patterns: Patterns indicating sensitive data.

    Returns:
        Configured APITarget.
    """
    system = TargetSystem(
        type=TargetType.API,
        name=name,
        endpoint=endpoint,
        interface="http",
        sensitive_patterns=sensitive_patterns or [],
    )
    return APITarget(system, method=method, headers=headers)


def create_prompt_target(
    name: str,
    endpoint: str,
    system_prompt: str,
    call_fn: Callable[[str, str], str],
    sensitive_patterns: list[str] | None = None,
) -> PromptTarget:
    """Factory function to create a prompt injection target.

    Args:
        name: Human-readable name.
        endpoint: Model identifier.
        system_prompt: System prompt to test.
        call_fn: Function taking (system_prompt, user_input).
        sensitive_patterns: Patterns indicating sensitive data.

    Returns:
        Configured PromptTarget.
    """
    system = TargetSystem(
        type=TargetType.PROMPT,
        name=name,
        endpoint=endpoint,
        interface="sdk",
        sensitive_patterns=sensitive_patterns or [],
    )
    return PromptTarget(system, system_prompt=system_prompt, call_fn=call_fn)
