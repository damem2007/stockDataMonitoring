from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import requests

from app.config import (
    AI_DISCLOSURE,
    LOCAL_DISCLOSURE,
    OPENAI_CONNECT_TIMEOUT_SECONDS,
    OPENAI_MAX_OUTPUT_TOKENS,
    OPENAI_REASONING_EFFORT,
    OPENAI_READ_TIMEOUT_SECONDS,
    OPENAI_RESPONSES_ENDPOINT,
)
from app.services.ai.models import resolve_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PROVIDER_OPENAI = "openai"
PROVIDER_QWEN = "qwen"
PROVIDER_LOCAL = "local-rules"


@dataclass
class ProviderRequest:
    """One structured-JSON interpretation request.

    ``prompt`` is serialized as the user message, ``schema`` is enforced with
    the provider's JSON-schema output mode, and ``log_prefix``/``logger`` let
    each calling service keep its own telemetry stream.
    """

    task: str
    prompt: dict[str, Any]
    schema: dict[str, Any]
    schema_name: str
    system: str = AI_DISCLOSURE
    strict: bool = False
    max_output_tokens: int = OPENAI_MAX_OUTPUT_TOKENS
    reasoning_effort: str | None = OPENAI_REASONING_EFFORT
    connect_timeout: int = OPENAI_CONNECT_TIMEOUT_SECONDS
    read_timeout: int = OPENAI_READ_TIMEOUT_SECONDS
    model: str | None = None
    log_prefix: str = "ai_provider"
    logger: logging.Logger | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResult:
    status: str
    provider: str
    model: str
    payload: dict[str, Any] | None = None
    reason: str | None = None
    failure_category: str | None = None
    elapsed_ms: int | None = None
    response_status: int | None = None
    request_bytes: int | None = None
    response_excerpt: str = ""
    attempts: list[dict[str, Any]] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def ok(self) -> bool:
        return self.status == "ready" and isinstance(self.payload, dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "provider": self.provider,
            "model": self.model,
            "reason": self.reason,
            "failureCategory": self.failure_category,
            "elapsedMs": self.elapsed_ms,
            "responseStatus": self.response_status,
            "requestBytes": self.request_bytes,
            "attempts": list(self.attempts),
            "generatedAt": self.generated_at,
        }


class AIProvider(Protocol):
    id: str
    label: str

    def configured(self) -> bool: ...

    def status(self) -> dict[str, Any]: ...

    def generate_json(self, request: ProviderRequest) -> ProviderResult: ...


def _log(request: ProviderRequest, level: int, message: str, *args: Any) -> None:
    (request.logger or logger).log(level, message, *args)


def _context_kv(request: ProviderRequest) -> str:
    return " ".join(f"{key}={value}" for key, value in request.context.items())


def extract_json_object(payload: Any) -> dict[str, Any] | None:
    """Pull the first JSON object out of a Responses API payload."""

    if not isinstance(payload, dict):
        return None
    texts: list[Any] = [payload.get("output_text")]
    for output in payload.get("output", []) or []:
        if not isinstance(output, dict):
            continue
        for content in output.get("content", []) or []:
            if isinstance(content, dict):
                texts.append(content.get("text") or content.get("output_text"))
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def redact(value: str, limit: int = 8000) -> str:
    return re.sub(r"Bearer\s+\S+", "Bearer [redacted]", str(value or ""))[:limit]


class OpenAIProvider:
    id = PROVIDER_OPENAI
    label = "OpenAI"

    def configured(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    def status(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "configured": self.configured(),
            "enabled": True,
            "role": "primary",
            "endpoint": OPENAI_RESPONSES_ENDPOINT,
            "note": "Active interpretation provider. Local calculations build the evidence and remain authoritative.",
        }

    def build_body(self, request: ProviderRequest, model: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": json.dumps(request.prompt, default=str)},
            ],
            "max_output_tokens": request.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": request.schema_name,
                    "strict": request.strict,
                    "schema": request.schema,
                }
            },
        }
        if model.startswith("gpt-5") and request.reasoning_effort:
            body["reasoning"] = {"effort": request.reasoning_effort}
        return body

    def generate_json(self, request: ProviderRequest) -> ProviderResult:
        model = request.model or resolve_model(request.task)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            _log(
                request,
                logging.WARNING,
                "%s provider=openai outcome=skipped reason=missing_api_key fallback=local-rules model=%s endpoint=%s %s",
                request.log_prefix,
                model,
                OPENAI_RESPONSES_ENDPOINT,
                _context_kv(request),
            )
            return ProviderResult(
                status="unavailable",
                provider=self.id,
                model=model,
                reason="OpenAI was not configured on the backend.",
                failure_category="configuration",
                request_bytes=0,
                elapsed_ms=0,
            )
        body = self.build_body(request, model)
        request_bytes = len(json.dumps(body, default=str).encode("utf-8"))
        started = time.monotonic()
        _log(
            request,
            logging.INFO,
            "%s provider=openai outcome=attempt method=POST model=%s endpoint=%s connect_timeout_seconds=%s "
            "read_timeout_seconds=%s max_output_tokens=%s request_bytes=%s %s",
            request.log_prefix,
            model,
            OPENAI_RESPONSES_ENDPOINT,
            request.connect_timeout,
            request.read_timeout,
            request.max_output_tokens,
            request_bytes,
            _context_kv(request),
        )
        try:
            response = requests.post(
                OPENAI_RESPONSES_ENDPOINT,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body,
                timeout=(request.connect_timeout, request.read_timeout),
            )
            elapsed_ms = round((time.monotonic() - started) * 1000)
            _log(
                request,
                logging.INFO,
                "%s provider=openai outcome=response model=%s status=%s elapsed_ms=%s %s",
                request.log_prefix,
                model,
                response.status_code,
                elapsed_ms,
                _context_kv(request),
            )
            response.raise_for_status()
            parsed = extract_json_object(response.json())
            if not isinstance(parsed, dict):
                return self._fallback(
                    request,
                    model,
                    "invalid_response",
                    "OpenAI returned an unusable analysis response.",
                    request_bytes,
                    started,
                    status=response.status_code,
                    excerpt=redact(response.text),
                )
            return ProviderResult(
                status="ready",
                provider=self.id,
                model=model,
                payload=parsed,
                elapsed_ms=elapsed_ms,
                response_status=response.status_code,
                request_bytes=request_bytes,
            )
        except requests.Timeout:
            return self._fallback(request, model, "timeout", f"OpenAI did not respond within {request.read_timeout} seconds.", request_bytes, started)
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else None
            excerpt = redact(error.response.text if error.response is not None else "")
            return self._fallback(request, model, "http_error", f"OpenAI returned HTTP {status if status is not None else 'unknown'}.", request_bytes, started, status=status, excerpt=excerpt)
        except requests.RequestException as error:
            return self._fallback(request, model, "request_error", "The OpenAI request could not be completed.", request_bytes, started, error_type=type(error).__name__)
        except (ValueError, TypeError) as error:
            return self._fallback(request, model, "parse_error", "OpenAI returned a response the application could not parse.", request_bytes, started, error_type=type(error).__name__)
        except Exception as error:  # pragma: no cover - defensive guard for provider surprises
            (request.logger or logger).exception("%s provider=openai outcome=unexpected_error model=%s", request.log_prefix, model)
            return self._fallback(request, model, "unexpected_error", "OpenAI failed unexpectedly while generating the analysis.", request_bytes, started, error_type=type(error).__name__)

    def _fallback(
        self,
        request: ProviderRequest,
        model: str,
        category: str,
        reason: str,
        request_bytes: int,
        started: float,
        *,
        status: int | None = None,
        excerpt: str = "",
        error_type: str | None = None,
    ) -> ProviderResult:
        elapsed_ms = round((time.monotonic() - started) * 1000)
        _log(
            request,
            logging.WARNING,
            "%s provider=openai outcome=fallback reason=%s fallback=local-rules model=%s endpoint=%s "
            "read_timeout_seconds=%s request_bytes=%s elapsed_ms=%s status=%s error_type=%s %s",
            request.log_prefix,
            category,
            model,
            OPENAI_RESPONSES_ENDPOINT,
            request.read_timeout,
            request_bytes,
            elapsed_ms,
            status if status is not None else "none",
            error_type or "none",
            _context_kv(request),
        )
        return ProviderResult(
            status="fallback",
            provider=self.id,
            model=model,
            reason=reason,
            failure_category=category,
            elapsed_ms=elapsed_ms,
            response_status=status,
            request_bytes=request_bytes,
            response_excerpt=excerpt,
        )


class QwenProvider:
    """Reserved slot for the in-house Qwen model.

    The adapter exposes its configuration surface (``QWEN_API_BASE_URL``,
    ``QWEN_MODEL``, ``QWEN_API_KEY``, ``AI_ENABLE_QWEN``) so deployment and
    monitoring can be wired ahead of time, but it does not issue requests yet.
    When the trained model is ready, ``generate_json`` should post the same
    ``ProviderRequest`` to the OpenAI-compatible chat endpoint the in-house
    serving stack exposes (``/v1/chat/completions`` with a JSON response
    format) and map the reply through :func:`extract_json_object`.
    """

    id = PROVIDER_QWEN
    label = "Qwen (in-house)"

    def configured(self) -> bool:
        return False

    def enabled(self) -> bool:
        return os.getenv("AI_ENABLE_QWEN", "false").strip().lower() == "true"

    def status(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "configured": bool(os.getenv("QWEN_API_BASE_URL")),
            "enabled": self.enabled(),
            "role": "reserved",
            "endpoint": os.getenv("QWEN_API_BASE_URL") or None,
            "model": os.getenv("QWEN_MODEL") or "qwen (not deployed)",
            "note": "Reserved provider slot for the in-house Qwen model. The adapter is wired but intentionally not active in this build.",
        }

    def generate_json(self, request: ProviderRequest) -> ProviderResult:
        _log(request, logging.INFO, "%s provider=qwen outcome=skipped reason=reserved_provider", request.log_prefix)
        return ProviderResult(
            status="unavailable",
            provider=self.id,
            model=os.getenv("QWEN_MODEL") or "qwen (not deployed)",
            reason="The in-house Qwen provider is reserved and not enabled in this build.",
            failure_category="not_enabled",
            elapsed_ms=0,
            request_bytes=0,
        )


class LocalRulesProvider:
    """Marker for the deterministic fallback so status pages can describe the chain."""

    id = PROVIDER_LOCAL
    label = "Local rules"

    def configured(self) -> bool:
        return True

    def status(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "configured": True,
            "enabled": True,
            "role": "fallback",
            "model": "deterministic local rules",
            "note": LOCAL_DISCLOSURE,
        }

    def generate_json(self, request: ProviderRequest) -> ProviderResult:
        return ProviderResult(status="unavailable", provider=self.id, model="deterministic local rules", reason="Local rules do not generate provider output; each service computes its own fallback.", failure_category="local")
