import logging
import time

from langchain_core.language_models.chat_models import BaseChatModel

from backend.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("gemini", "groq")

# Track provider health status
_provider_status: dict[str, dict] = {
    "gemini": {"healthy": True, "last_error": None, "error_count": 0, "last_check": 0.0},
    "groq": {"healthy": True, "last_error": None, "error_count": 0, "last_check": 0.0},
}

FALLBACK_ORDER = {"gemini": "groq", "groq": "gemini"}


def get_provider_model(provider: str) -> str:
    provider = provider.lower()
    if provider == "gemini":
        return settings.gemini_model
    if provider == "groq":
        return settings.groq_model
    raise ValueError(f"Unknown LLM provider: {provider}. Use 'gemini' or 'groq'.")


def is_provider_configured(provider: str) -> bool:
    provider = provider.lower()
    if provider == "gemini":
        return bool(settings.gemini_api_key.strip())
    if provider == "groq":
        return bool(settings.groq_api_key.strip())
    return False


def get_available_llm_options() -> list[dict[str, str | bool]]:
    return [
        {
            "provider": provider,
            "label": provider.title(),
            "model": get_provider_model(provider),
            "enabled": is_provider_configured(provider),
        }
        for provider in SUPPORTED_PROVIDERS
    ]


def get_llm(provider: str | None = None) -> BaseChatModel:
    provider = (provider or settings.llm_provider).lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        logger.info("Using Gemini model: %s", settings.gemini_model)
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        logger.info("Using Groq model: %s", settings.groq_model)
        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0,
        )

    raise ValueError(f"Unknown LLM provider: {provider}. Use 'gemini' or 'groq'.")


def mark_provider_error(provider: str, error: str) -> None:
    """Mark a provider as having an error (for fallback decisions)."""
    provider = provider.lower()
    if provider in _provider_status:
        status = _provider_status[provider]
        status["error_count"] += 1
        status["last_error"] = error
        status["last_check"] = time.time()
        if status["error_count"] >= 3:
            status["healthy"] = False
            logger.warning("Provider %s marked unhealthy after %d errors", provider, status["error_count"])


def mark_provider_healthy(provider: str) -> None:
    """Mark a provider as healthy after successful use."""
    provider = provider.lower()
    if provider in _provider_status:
        _provider_status[provider]["healthy"] = True
        _provider_status[provider]["error_count"] = 0
        _provider_status[provider]["last_error"] = None
        _provider_status[provider]["last_check"] = time.time()


def get_fallback_provider(provider: str) -> str | None:
    """Get the fallback provider if the primary is unhealthy."""
    provider = provider.lower()
    fallback = FALLBACK_ORDER.get(provider)
    if fallback and is_provider_configured(fallback):
        return fallback
    return None


def get_provider_status() -> dict[str, dict]:
    """Get health status for all providers."""
    return {
        provider: {
            "healthy": status["healthy"],
            "configured": is_provider_configured(provider),
            "model": get_provider_model(provider) if is_provider_configured(provider) else None,
            "error_count": status["error_count"],
            "last_error": status["last_error"],
        }
        for provider, status in _provider_status.items()
    }
