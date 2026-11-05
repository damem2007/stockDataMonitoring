from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


# Environment-backed settings live here so provider changes do not require
# hunting through the analysis services. Defaults keep local development
# operable when a non-secret setting is omitted.
AI_DISCLOSURE = os.getenv(
    "AI_DISCLOSURE",
    "OpenAI-assisted investment research is being used. This is educational decision support, not financial advice.",
)
LOCAL_DISCLOSURE = os.getenv(
    "LOCAL_DISCLOSURE",
    "Deterministic local calculations are being used as the final fallback. This is educational decision support, not financial advice.",
)
RECOMMENDED_OPENAI_MODEL = os.getenv("RECOMMENDED_OPENAI_MODEL", "gpt-5.6-sol")
OPENAI_RESPONSES_ENDPOINT = os.getenv("OPENAI_RESPONSES_ENDPOINT", "https://api.openai.com/v1/responses")
OPENAI_CONNECT_TIMEOUT_SECONDS = _env_int("OPENAI_CONNECT_TIMEOUT_SECONDS", 10)
OPENAI_READ_TIMEOUT_SECONDS = _env_int("OPENAI_READ_TIMEOUT_SECONDS", 90)
OPENAI_MAX_OUTPUT_TOKENS = _env_int("OPENAI_MAX_OUTPUT_TOKENS", 1000)
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low")
REFRESH_INTERVAL_SECONDS = max(5, _env_int("REFRESH_INTERVAL_SECONDS", 30))
FUTURE_PROVIDER_NOTE = os.getenv(
    "FUTURE_PROVIDER_NOTE",
    "OpenAI is the preferred interpretation provider. Local calculations build and validate the evidence baseline; deterministic local rules are the final fallback. Qwen is reserved as a future provider slot.",
)
MARKET_EXPLORER_RESULT_SCHEMA_VERSION = os.getenv("MARKET_EXPLORER_RESULT_SCHEMA_VERSION", "market-explorer.v1")

PUBLIC_HOLIDAY_COUNTRIES = tuple(
    code.strip().upper() for code in os.getenv("PUBLIC_HOLIDAY_COUNTRIES", "CA,US").split(",") if code.strip()
)
PUBLIC_HOLIDAY_SOURCE_URL = os.getenv(
    "PUBLIC_HOLIDAY_SOURCE_URL",
    "https://date.nager.at/api/v3/PublicHolidays/{year}/{country}",
)
PUBLIC_HOLIDAY_SYNC_YEARS = _env_int("PUBLIC_HOLIDAY_SYNC_YEARS", 2)
