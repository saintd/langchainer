import logging
import os  # Import os for example usage in docstring
from typing import Optional, TypedDict

from langchain_core.exceptions import LangChainException
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.utils import Input, Output

from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel


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
    rate_limiter: InMemoryRateLimiter | None
    base_url: str


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
        base_url: str | None = None,
) -> BaseChatModel | Runnable[Input, Output]:
    """
    Initializes a language model using LangChain's `init_chat_model` and
    optionally wraps it with retry logic.

    This function is a wrapper around LangChain's `init_chat_model`, providing
    additional functionality such as rate limiting, and retry logic.

    Args:
        model: The model identifier string.  This can be specified in several ways:

            1. **Provider/Model:**  A string with the format
               `"provider/model_name"` (e.g., `"openai/gpt-3.5-turbo"`,
               `"mistralai/mistral-small-latest"`).  This is the most common
               and recommended format for most providers.

            2. **Model Name Only:** Just the model name (e.g., `"gpt-3.5-turbo"`,
               `"mistral-small-latest"`).  If only the model name is provided,
               the function will attempt to *infer* the provider based on
               common prefixes (see "Provider Inference" below).

            3. **Google GenAI with `genai:` prefix:**  For Google's GenAI models
               (using `langchain-google-genai`), use the prefix `genai:`
               followed by the model name (e.g., `"genai:gemini-pro"`).  This
               explicitly selects the `langchain-google-genai` provider.  Using
               the `genai:` prefix is equivalent to providing no prefix and
               just the model name when the model name starts with "gemini".

            4. **OpenRouter Models with `openrouter/` prefix:** For models
               hosted on OpenRouter.ai, use the prefix `openrouter/` followed
               by the OpenRouter model path (e.g.,
               `"openrouter/google/gemini-2.0-flash-exp:free"`,
               `"openrouter/mistralai/mistral-medium"`).  This automatically
               sets the `base_url` to OpenRouter's API endpoint and uses the
               `langchain-openai` integration (as OpenRouter has an
               OpenAI-compatible API).

            **Provider Inference (when only the model name is given):**

            If you provide *only* the model name, the function attempts to
            infer the provider based on these prefixes:

            *   `"gpt-3..."`, `"gpt-4..."`, `"o1..."`:  `openai`
            *   `"claude..."`:  `anthropic`
            *   `"gemini..."`: `google_genai` (default for Gemini)
            *   `"command..."`:  `cohere`
            *   `"accounts/fireworks..."`:  `fireworks`
            *   `"mistral..."`:  `mistralai`

            **Important Notes:**

            *   **`base_url` for Direct Provider Access (not OpenRouter):** When
                using the `base_url` parameter to connect to an
                OpenAI-compatible provider *directly* (not via the
                `openrouter/` prefix), you *must* use the `"openai/"` prefix in
                your model name, even if it's not an OpenAI model.  Example:
                `model="openai/my-custom-model", base_url="https://...your-api..."`.
                This ensures the correct Langchain integration is used.  If
                using `openrouter/`, `base_url` is handled automatically.

            *   **OpenRouter `api_key`:** When using the `openrouter/` prefix,
                the `api_key` argument should be your OpenRouter API key.

        base_url:  Optional. A base URL for the provider.
            *   If using `openrouter/`, this is automatically set and any
                provided `base_url` is ignored.
            *   For direct access to OpenAI-compatible providers, set this to
                the provider's API endpoint.
            *   If not using `openrouter/` and not providing a custom
                `base_url`, the default base URL for the inferred or specified
                provider will be used.

        temperature: Optional. The temperature to use for sampling (0.0 for
            deterministic, higher for more creative).
        max_tokens: Optional. The maximum number of tokens to generate.
        api_key: Optional.  Your API key for the model provider.
        requests_per_second: Optional.  Rate limiting: maximum requests per
            second.  Provide either `requests_per_second` *or* `rate_limiter`,
            but not both.
        rate_limiter: Optional.  A custom `InMemoryRateLimiter` instance.  If
            both `requests_per_second` and `rate_limiter` are provided,
            `rate_limiter` takes precedence.
        retry_exception_types: Optional.  A tuple of exception types to retry
            on (e.g., `(ConnectionError,)`).
        wait_exponential_jitter: Optional.  Use exponential backoff with jitter
            when retrying (default: True).
        stop_after_attempt: Optional. Maximum number of retry attempts (default:
            3).
        return_runnable: Optional.  If True, return a `Runnable` with retry
            logic.  If False (or if no retry options are set), return a
            `BaseChatModel`.
        stop_after_attempt: The maximum number of retry attempts.
        return_runnable: Whether to return a runnable with retry logic.

    Returns:
        An initialized `BaseChatModel` or a `Runnable` with retry logic (if
        `return_runnable` is True or retry options are set).

        `Input` represents the input type (e.g., `PromptValue`, `str`,
        `List[ChatMessage]`) and `Output` represents the output type (e.g.,
        `BaseMessage`, `str`) of the language model.

    Raises:
        MissingModelNameError: If the `model` argument is empty.
        LLMInitializationError:  If there's an error initializing the model.
        LLMProviderNotInstalledError: If the required provider package isn't
            installed (e.g., `langchain-openai`, `langchain-google-genai`).

    Note:
        This function uses LangChain's `init_chat_model` internally. The following
        model providers are supported, and the corresponding integration package
        must be installed:

        - 'openai' -> langchain-openai (also used for OpenRouter)
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

    Examples:
        ```python
        from langchain_core.rate_limiters import InMemoryRateLimiter
        import os

        # Standard OpenAI model
        llm_openai = init_llm("openai/gpt-3.5-turbo", api_key="YOUR_OPENAI_KEY")

        # Google GenAI model (explicit provider via prefix)
        llm_genai_prefix = init_llm("genai:gemini-pro", api_key=os.getenv('GEMINI_API_KEY'))

        # Google GenAI model (default provider, no prefix needed)
        llm_genai = init_llm("gemini-pro", api_key=os.getenv('GEMINI_API_KEY'))

        # Google VertexAI model (explicit provider)
        llm_vertex = init_llm("google_vertexai/gemini-pro", api_key="YOUR_VERTEX_API_KEY")

        # OpenRouter model (automatic base_url)
        llm_openrouter = init_llm("openrouter/mistralai/mistral-medium", api_key="YOUR_OPENROUTER_KEY")
        llm_openrouter_gemini = init_llm("openrouter/google/gemini-2.0-flash-exp:free", api_key="YOUR_OPENROUTER_KEY")

        # Rate-limited model with retry
        rate_limiter = InMemoryRateLimiter(requests_per_second=5)
        llm_with_retry = init_llm(
            "openai/gpt-4",
            temperature=0.7,
            api_key="YOUR_OPENAI_KEY",
            rate_limiter=rate_limiter,
            retry_exception_types=(ConnectionError,),
            stop_after_attempt=5,
            return_runnable=True,
        )
        ```
    """
    if not model:
        raise MissingModelNameError("Model path cannot be empty")

    # Initialize provider and model_name, and handle prefixes
    provider = None
    model_name = model

    parts = model.split("/")
    if len(parts) > 1:
        provider, model_name = parts[0], "/".join(parts[1:])
    else:
        provider, model_name = None, model

    # Add OpenRouter special case FIRST
    if provider == "openrouter":
        # Set base URL for OpenRouter's API
        base_url = "https://openrouter.ai/api/v1"
        # OpenRouter uses OpenAI-compatible API - force provider to 'openai'
        provider = "openai"
    elif model.startswith("genai:"):  # Handle genai: prefix
        provider = "google_genai"
        model_name = model[len("genai:"):]
    # Then handle other provider inferences
    elif provider is None:
        if model_name.startswith(("gpt-3", "gpt-4", "o1")):
            provider = "openai"
        elif model_name.startswith("claude"):
            provider = "anthropic"
        elif model_name.startswith("gemini"):
            provider = "google_genai"  # Default to GenAI for Gemini
        elif model_name.startswith("command"):
            provider = "cohere"
        elif model_name.startswith("accounts/fireworks"):
            provider = "fireworks"
        elif model_name.startswith("mistral"):
            provider = "mistralai"

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
    if base_url is not None:  # base_url is set automatically for openrouter
        init_kwargs["base_url"] = base_url

    if requests_per_second is not None and rate_limiter is not None:
        logging.warning(
            "Both 'requests_per_second' and 'rate_limiter' were provided. 'rate_limiter' takes precedence."
        )

    if rate_limiter is None and requests_per_second is not None:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=max(1, int(1 / requests_per_second))
            if requests_per_second > 0
            else 1,
            max_bucket_size=max(1, int(requests_per_second)),
        )

    if rate_limiter is not None:
        init_kwargs["rate_limiter"] = rate_limiter

    try:
        llm = init_chat_model(model_name, model_provider=provider, **init_kwargs)

        # Handle default values for retry arguments
        should_apply_retry = (
                retry_exception_types is not None
                or wait_exponential_jitter is not None
                or stop_after_attempt is not None
        )

        if return_runnable or should_apply_retry:
            # Use default values if arguments are None
            retry_exception_types = retry_exception_types or (Exception,)
            wait_exponential_jitter = (
                wait_exponential_jitter if wait_exponential_jitter is not None else True
            )
            stop_after_attempt = stop_after_attempt or 3

            return llm.with_retry(
                retry_if_exception_type=retry_exception_types,
                wait_exponential_jitter=wait_exponential_jitter,
                stop_after_attempt=stop_after_attempt,
            )
        else:
            return llm

    except ValueError as e:
        raise LLMInitializationError(f"Error initializing model: {e}") from e
    except ImportError as e:
        raise LLMProviderNotInstalledError(
            f"Error importing model or dependencies. Ensure that model provider integration package is installed: {e}"
        ) from e
    except Exception as e:
        raise LLMInitializationError(f"An unexpected error occurred during model initialization: {e}") from e
