"""
Relax AI LLM Client for UK LawAssistant.

Async HTTP client that communicates with the Relax AI API (OpenAI-compatible).
Handles:
- Chat completions (single turn and multi-turn)
- Streaming responses
- Automatic retries with exponential backoff
- Structured error handling
- Token usage tracking

This is the sole interface between our system and the LLM.
Every other module talks to this client — never directly to the API.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from app.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A single message in a chat conversation."""
    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class TokenUsage:
    """Token counts from an LLM response."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """Structured response from the LLM."""
    content: str
    model: str
    usage: TokenUsage
    latency_ms: float
    finish_reason: str = "stop"
    raw_response: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RelaxAIError(Exception):
    """Base exception for Relax AI client errors."""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class RelaxAIRateLimitError(RelaxAIError):
    """Raised when we hit the API rate limit (429)."""
    pass


class RelaxAIServerError(RelaxAIError):
    """Raised on 5xx server errors (retriable)."""
    pass


class RelaxAIAuthError(RelaxAIError):
    """Raised on 401/403 authentication errors (not retriable)."""
    pass


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class RelaxAIClient:
    """
    Async client for Relax AI's OpenAI-compatible chat completions API.

    Usage:
        async with RelaxAIClient() as client:
            response = await client.chat("What is the Data Protection Act?")
            print(response.content)
    """

    def __init__(self):
        settings = get_settings().relax_ai
        self._base_url = settings.base_url
        self._model = settings.model
        self._max_tokens = settings.max_tokens
        self._temperature = settings.temperature
        self._timeout = settings.timeout
        self._max_retries = settings.max_retries

        self._headers = {
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
        }

        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "RelaxAIClient":
        """Open the HTTP connection pool."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=httpx.Timeout(self._timeout, connect=10.0),
        )
        return self

    async def __aexit__(self, *args) -> None:
        """Close the HTTP connection pool."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure the client is initialized."""
        if self._client is None:
            raise RuntimeError(
                "RelaxAIClient must be used as an async context manager: "
                "'async with RelaxAIClient() as client: ...'"
            )
        return self._client

    # ------------------------------------------------------------------
    # Core API call with retries
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((RelaxAIRateLimitError, RelaxAIServerError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _call_api(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """
        Make a chat completion request to Relax AI.

        Retries automatically on rate limits and server errors.
        Fails immediately on auth errors or bad requests.
        """
        client = self._ensure_client()

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }

        try:
            response = await client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as e:
            raise RelaxAIServerError(
                f"Request timed out after {self._timeout}s", status_code=408
            ) from e
        except httpx.ConnectError as e:
            raise RelaxAIServerError(
                f"Connection failed: {e}", status_code=503
            ) from e

        # Handle error responses
        if response.status_code == 429:
            raise RelaxAIRateLimitError(
                "Rate limit exceeded", status_code=429
            )
        elif response.status_code in (401, 403):
            raise RelaxAIAuthError(
                f"Authentication failed: {response.text}", status_code=response.status_code
            )
        elif response.status_code >= 500:
            raise RelaxAIServerError(
                f"Server error: {response.text}", status_code=response.status_code
            )
        elif response.status_code != 200:
            raise RelaxAIError(
                f"API error ({response.status_code}): {response.text}",
                status_code=response.status_code,
            )

        return response.json()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def chat(
        self,
        user_message: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Send a single question to the LLM and get a response.

        Args:
            user_message: The user's question or prompt.
            system_prompt: Optional system prompt to set behavior.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            LLMResponse with content, usage stats, and latency.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        return await self.chat_messages(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def chat_messages(
        self,
        messages: list[dict | Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Send a multi-turn conversation to the LLM.

        Args:
            messages: List of message dicts or Message objects.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            LLMResponse with content, usage stats, and latency.
        """
        # Normalize Message objects to dicts
        normalized = []
        for msg in messages:
            if isinstance(msg, Message):
                normalized.append({"role": msg.role, "content": msg.content})
            else:
                normalized.append(msg)

        start = time.perf_counter()
        raw = await self._call_api(
            messages=normalized,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        # Parse response
        choice = raw.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage_data = raw.get("usage", {})

        # DeepSeek models may return reasoning in "reasoning_content"
        # and the actual answer in "content". Handle both cases.
        content = message.get("content") or ""
        if not content.strip():
            content = message.get("reasoning_content") or ""

        return LLMResponse(
            content=content.strip(),
            model=raw.get("model", self._model),
            usage=TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
            latency_ms=round(latency_ms, 2),
            finish_reason=choice.get("finish_reason", "unknown"),
            raw_response=raw,
        )

    async def stream(
        self,
        user_message: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a response from the LLM token-by-token.

        Yields content chunks as they arrive. Useful for real-time UI.

        Usage:
            async with RelaxAIClient() as client:
                async for chunk in client.stream("Explain GDPR"):
                    print(chunk, end="", flush=True)
        """
        client = self._ensure_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "stream": True,
        }

        async with client.stream("POST", "/chat/completions", json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise RelaxAIError(
                    f"Stream error ({response.status_code}): {body.decode()}",
                    status_code=response.status_code,
                )

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]  # Strip "data: " prefix
                if data.strip() == "[DONE]":
                    break

                import json
                try:
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue
