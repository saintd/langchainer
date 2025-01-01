
from typing import Optional, TypedDict

from langchain_core.exceptions import LangChainException
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.utils import Input, Output

from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel

# # We used to need _ConfigurableModel to have its type hint in the return of the function
# # but now, since we always use `model` as not None, we don't need it.
# from langchain_core.language_models.base import LanguageModelInput
# # Define _ConfigurableModel as a workaround for importing a private class
# _ConfigurableModel = Runnable[LanguageModelInput, Any]

class MissingModelNameError(LangChainException):
    """Raised when the model name (identifier) is missing or empty."""
    pass

class LLMInitializationError(LangChainException):
    """Raised when there is an error initializing an LLM."""
    pass

class LLMProviderNotInstalledError(LangChainException):
    """Raised when the required provider package is not installed."""
    pass

class InitKwargs(TypedDict, total=False):
    temperature: float
    max_tokens: int
    api_key: str
    rate_limiter: Optional[InMemoryRateLimiter]


def init_llm(
        model: str,
        *,
        temperature: float | None = 0,
        max_tokens: int | None = None,
        api_key: str | None = None,
        requests_per_second: float | None = None,
        rate_limiter: InMemoryRateLimiter | None = None,
        retry_exception_types: tuple[type[BaseException], ...] | None = None,
        wait_exponential_jitter: bool | None = None,
        stop_after_attempt: int | None = None,
        return_runnable: bool | None = None,
    ) -> BaseChatModel | Runnable[Input, Output]:
    """
    Initializes a language model using LangChain's `init_chat_model` and
    optionally wraps it with retry logic.

    This function is a wrapper around LangChain's `init_chat_model`, providing additional
    functionality such as rate limiting, and retry logic.

    Args:
        model: The model identifier string. This can be:
               1. A full provider/model string (e.g., "openai/gpt-3.5-turbo", "mistralai/mistral-small-latest")
               2. Just the model name (e.g., "gpt-3.5-turbo", "mistral-small-latest")
               If only the model name is provided, the function will attempt to
               infer the provider automatically based on the model prefix.
               Supported prefixes for auto-inference include:
               - 'gpt-3...', 'gpt-4...', 'o1...' -> 'openai'
               - 'claude...' -> 'anthropic'
               - 'amazon....' -> 'bedrock'
               - 'gemini...' -> 'google_vertexai'
               - 'command...' -> 'cohere'
               - 'accounts/fireworks...' -> 'fireworks'
               - 'mistral...' -> 'mistralai'
        temperature: The temperature to use for sampling.
        max_tokens: The maximum number of tokens to generate.
        api_key: An optional API key for the model.
        requests_per_second: The maximum number of requests per second to allow.
                              If provided, a rate limiter will be created.
                              Note: It is recommended to provide either
                              `requests_per_second` or `rate_limiter`, but not both.
        rate_limiter: An optional rate limiter to use.
                      If both requests_per_second and rate_limiter are provided,
                      rate_limiter takes precedence.
        retry_exception_types: A tuple of exception types to retry on.
        wait_exponential_jitter: Whether to use exponential backoff with jitter
                                 when retrying.
        stop_after_attempt: The maximum number of retry attempts.
        return_runnable: Whether to return a runnable with retry logic.

    Returns:
        An initialized BaseChatModel. If `return_runnable` is True or if any of
        `retry_exception_types`, `wait_exponential_jitter`, or `stop_after_attempt`
        are provided, a Runnable with retry logic is returned instead.

    Raises:
        MissingModelNameError: If the `model` argument is empty or not provided.
        LLMInitializationError: If there is an error initializing the model.
        LLMProviderNotInstalledError: If the required provider package is not installed.

    Note:
        This function uses LangChain's `init_chat_model` internally. The following
        model providers are supported, and the corresponding integration package
        must be installed:

        - 'openai' -> langchain-openai
        - 'anthropic' -> langchain-anthropic
        - 'azure_openai' -> langchain-openai
        - 'google_vertexai' -> langchain-google-vertexai
        - 'google_genai' -> langchain-google-genai
        - 'bedrock' -> langchain-aws
        - 'bedrock_converse' -> langchain-aws
        - 'cohere' -> langchain-cohere
        - 'fireworks' -> langchain-fireworks
        - 'together' -> langchain-together
        - 'mistralai' -> langchain-mistralai
        - 'huggingface' -> langchain-huggingface
        - 'groq' -> langchain-groq
        - 'ollama' -> langchain-ollama

        The function will attempt to infer the model provider if not explicitly specified.
    """
    if not model:
        raise MissingModelNameError("Model path cannot be empty")

    # Split the model string, but don't assume a provider is always specified
    parts = model.split("/")
    if len(parts) > 1:
        provider, model_name = parts[0], "/".join(parts[1:])
    else:
        provider, model_name = None, model

    # Handle special cases or aliases (like "mistral" to "mistralai")
    if provider == "mistral":
        provider = "mistralai"

    init_kwargs: InitKwargs = {}
    if temperature is not None:
        init_kwargs["temperature"] = temperature
    if max_tokens is not None:
        init_kwargs["max_tokens"] = max_tokens
    if api_key is not None:
        init_kwargs["api_key"] = api_key


    if requests_per_second is not None and rate_limiter is not None:
        print("Warning: Both 'requests_per_second' and 'rate_limiter' were provided. 'rate_limiter' takes precedence.")

    if rate_limiter is None and requests_per_second is not None:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=max(1, int(1 / requests_per_second)) if requests_per_second > 0 else 1,
            max_bucket_size=max(1, int(requests_per_second)),
        )

    if rate_limiter is not None:
        init_kwargs["rate_limiter"] = rate_limiter

    try:
        llm = init_chat_model(
            model_name,
            model_provider=provider,
            **init_kwargs
        )

        # Correctly handle default values for retry arguments
        should_apply_retry = (
                retry_exception_types is not None
                or wait_exponential_jitter is not None
                or stop_after_attempt is not None
        )

        if return_runnable or should_apply_retry:
            # Use default values if arguments are None
            retry_exception_types = retry_exception_types or (Exception,)
            wait_exponential_jitter = wait_exponential_jitter if wait_exponential_jitter is not None else True
            stop_after_attempt = stop_after_attempt or 3

            return llm.with_retry(
                retry_if_exception_type=retry_exception_types,
                wait_exponential_jitter=wait_exponential_jitter,
                stop_after_attempt=stop_after_attempt
            )
        else:
            return llm

    except ValueError as e:
        raise LLMInitializationError(f"Error initializing model: {e}") from e
    except ImportError as e:
        raise LLMProviderNotInstalledError(f"Error importing model or dependencies. Ensure that model provider integration package is installed: {e}") from e
    except Exception as e:
        raise LLMInitializationError(f"An unexpected error occurred during model initialization: {e}") from e
