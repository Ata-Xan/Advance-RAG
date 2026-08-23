import re
import threading
import time
from collections.abc import Callable
from typing import TypeVar

import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings

BATCH_SIZE = 50
GEMINI_MODEL = "models/gemini-embedding-2-preview"
GEMINI_DIM = 3072
GEMINI_REQUESTS_PER_MINUTE = 100
MIN_REQUEST_INTERVAL_SECONDS = 60 / GEMINI_REQUESTS_PER_MINUTE
MAX_RATE_LIMIT_WAIT_SECONDS = 10 * 60
RATE_LIMIT_BUFFER_SECONDS = 2

_active_model: GoogleGenerativeAIEmbeddings | None = None
_request_lock = threading.Lock()
_last_request_at = 0.0
Result = TypeVar("Result")


def _init() -> None:
    """Initialise the sole embedding model used by this RAG collection."""
    global _active_model
    if _active_model is not None:
        return

    try:
        model = GoogleGenerativeAIEmbeddings(
            model=GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
        )
        _call_with_rate_limit_retry(lambda: model.embed_query("probe"))
    except Exception as error:
        logfire.error(f"Gemini embedding probe failed: {error}")
        raise RuntimeError(
            "Gemini embeddings are unavailable. This collection requires "
            f"{GEMINI_MODEL} ({GEMINI_DIM} dimensions), so no fallback model is used."
        ) from error

    _active_model = model
    logfire.info(f"Gemini embeddings ready ({GEMINI_MODEL}, {GEMINI_DIM}-dim).")


def get_embedding_dim() -> int:
    """Return the fixed vector dimension for the collection embedding model."""
    return GEMINI_DIM


def get_embedding_model_name() -> str:
    """Return the model recorded with each indexed point."""
    return GEMINI_MODEL


def _is_rate_limit(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in ("429", "rate", "quota", "resource_exhausted"))


def _retry_delay_seconds(error: Exception, attempt: int) -> float:
    """Honor Gemini's RetryInfo delay when present, then add a safety buffer."""
    message = str(error)
    match = re.search(r"(?:retrydelay|retry in)[^0-9]*(\d+(?:\.\d+)?)s", message, re.IGNORECASE)
    if match:
        return float(match.group(1)) + RATE_LIMIT_BUFFER_SECONDS
    return min(2**attempt, 60)


def _wait_for_request_slot() -> None:
    """Keep this process below the configured Gemini request rate."""
    global _last_request_at
    with _request_lock:
        wait = MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
        if wait > 0:
            logfire.info(f"Pacing Gemini embedding request for {wait:.2f}s.")
            time.sleep(wait)
        _last_request_at = time.monotonic()


def _call_with_rate_limit_retry(operation: Callable[[], Result]) -> Result:
    deadline = time.monotonic() + MAX_RATE_LIMIT_WAIT_SECONDS
    attempt = 0

    while True:
        _wait_for_request_slot()
        try:
            return operation()
        except Exception as error:
            if not _is_rate_limit(error):
                logfire.error(f"Gemini embedding failed: {error}")
                raise

            wait = _retry_delay_seconds(error, attempt)
            if time.monotonic() + wait > deadline:
                raise RuntimeError(
                    "Gemini rate limit did not clear within "
                    f"{MAX_RATE_LIMIT_WAIT_SECONDS} seconds."
                ) from error

            attempt += 1
            logfire.warning(
                f"Gemini rate limit hit; retrying in {wait:.2f}s "
                f"(attempt {attempt})."
            )
            time.sleep(wait)


def embed_query(query: str) -> list[float]:
    _init()
    return _call_with_rate_limit_retry(lambda: _active_model.embed_query(query))


def _embed_batch(batch: list[str]) -> list[list[float]]:
    return _call_with_rate_limit_retry(lambda: _active_model.embed_documents(batch))


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        with logfire.span("Embed batch", model=GEMINI_MODEL, start=start, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings