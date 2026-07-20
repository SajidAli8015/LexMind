"""
LLM Client for LexMind
Provider-agnostic LLM interface supporting Google Gemini,
OpenAI, Anthropic, and Azure OpenAI.

Usage:
    from src.llm_client import get_llm
    llm = get_llm()
    response = llm.invoke("Your question here")
    print(response.content)
"""

from typing import Optional
from src.config import settings
from loguru import logger


def get_llm(
    provider: str = None,
    temperature: float = None,
    max_tokens: int = None,
):
    """
    Get an LLM instance for the specified provider.

    Args:
        provider:    Which AI provider to use.
                     Options: google, openai, anthropic, azure
                     Defaults to LLM_PROVIDER in .env
        temperature: Response randomness (0=factual, 1=creative)
                     Defaults to LLM_TEMPERATURE in .env
        max_tokens:  Maximum response length in tokens
                     Defaults to LLM_MAX_TOKENS in .env

    Returns:
        A LangChain-compatible chat model object

    Raises:
        ValueError: If provider is not supported or
                    required credentials are missing
    """
    provider = (provider or settings.get_llm_provider()).lower()
    temperature = (
        temperature if temperature is not None
        else settings.LLM_TEMPERATURE
    )
    max_tokens = (
        max_tokens if max_tokens is not None
        else settings.LLM_MAX_TOKENS
    )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not settings.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not set in .env. "
                "Add your Google API key to .env file."
            )
        logger.info(
            f"Using Google Gemini | model: {settings.GEMINI_MODEL}"
        )
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not set in .env. "
                "Add your OpenAI API key to .env file."
            )
        logger.info(
            f"Using OpenAI | model: {settings.OPENAI_MODEL}"
        )
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY not set in .env. "
                "Add your Anthropic API key to .env file."
            )
        logger.info(
            f"Using Anthropic | model: {settings.ANTHROPIC_MODEL}"
        )
        return ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "azure":
        from langchain_openai import AzureChatOpenAI
        if not settings.AZURE_OPENAI_API_KEY:
            raise ValueError(
                "AZURE_OPENAI_API_KEY not set in .env. "
                "Add your Azure OpenAI key to .env file."
            )
        if not settings.AZURE_OPENAI_ENDPOINT:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT not set in .env. "
                "Add your Azure endpoint URL to .env file."
            )
        if not settings.AZURE_OPENAI_DEPLOYMENT:
            raise ValueError(
                "AZURE_OPENAI_DEPLOYMENT not set in .env. "
                "Add your deployment name to .env file."
            )
        logger.info(
            f"Using Azure OpenAI | "
            f"deployment: {settings.AZURE_OPENAI_DEPLOYMENT} | "
            f"endpoint: {settings.AZURE_OPENAI_ENDPOINT} | "
            f"api_version: {settings.AZURE_OPENAI_API_VERSION}"
        )
        return AzureChatOpenAI(
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(
        f"Unsupported provider '{provider}'. "
        f"Choose from: google, openai, anthropic, azure"
    )


def get_available_providers() -> dict:
    """
    Check which LLM providers are configured and ready to use.

    Returns:
        Dict with provider names as keys, each containing:
            configured (bool): True if API credentials are set
            model (str):       The model name that will be used

    Example return value:
        {
            'google':    {'configured': False, 'model': 'gemini-2.0-flash'},
            'openai':    {'configured': False, 'model': 'gpt-4o'},
            'anthropic': {'configured': False, 'model': 'claude-sonnet-4-5'},
            'azure':     {'configured': True,  'model': 'my-deployment'},
        }
    """
    return {
        "google": {
            "configured": bool(settings.GOOGLE_API_KEY),
            "model": settings.GEMINI_MODEL,
        },
        "openai": {
            "configured": bool(settings.OPENAI_API_KEY),
            "model": settings.OPENAI_MODEL,
        },
        "anthropic": {
            "configured": bool(settings.ANTHROPIC_API_KEY),
            "model": settings.ANTHROPIC_MODEL,
        },
        "azure": {
            "configured": bool(
                settings.AZURE_OPENAI_API_KEY
                and settings.AZURE_OPENAI_ENDPOINT
                and settings.AZURE_OPENAI_DEPLOYMENT
            ),
            "model": (
                settings.AZURE_OPENAI_DEPLOYMENT
                if settings.AZURE_OPENAI_DEPLOYMENT
                else "not configured"
            ),
        },
    }


def test_llm_connection(provider: Optional[str] = None) -> bool:
    """
    Test connectivity to an LLM provider.

    Sends a simple test message and checks for a response.
    Returns True if connection works, False if any error occurs.

    Args:
        provider: Provider to test (google/openai/anthropic/azure)
                  Defaults to LLM_PROVIDER in .env

    Returns:
        True if connection successful, False otherwise
    """
    provider = (provider or settings.get_llm_provider()).lower()
    try:
        llm = get_llm(provider=provider)
        response = llm.invoke("Say OK")
        logger.info(
            f"Connection test passed for '{provider}': "
            f"{response.content!r}"
        )
        return True
    except Exception as exc:
        logger.error(
            f"Connection test failed for '{provider}': {exc}"
        )
        return False
