"""LLM clients routed through the Portkey gateway."""

from langchain_openai import ChatOpenAI

from app.config import settings


def create_groq_llm(*, temperature: float) -> ChatOpenAI:
    """Create a Groq chat model using the managed ``@rag1`` Portkey provider."""
    if not settings.PORTKEY_API_KEY:
        raise RuntimeError("PORTKEY_API_KEY must be set to call the Portkey gateway.")

    return ChatOpenAI(
        # ChatOpenAI requires an OpenAI-compatible API key. Portkey authenticates
        # the request using x-portkey-api-key below, so this value is never sent to
        # Groq or used as a Groq credential.
        api_key="portkey-managed-provider",
        base_url=settings.PORTKEY_GATEWAY_URL,
        default_headers={
            "x-portkey-api-key": settings.PORTKEY_API_KEY,
            "x-portkey-provider": f"@{settings.PORTKEY_GROQ_PROVIDER}",
        },
        model=settings.GROQ_MODEL,
        temperature=temperature,
        model_kwargs={
            "reasoning_effort": "none",
        },
    )
