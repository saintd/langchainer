"""
This module provides a convenient interface for interacting with LLMs utilizing the LangChain library.

Key Features:
- Supports structured and unstructured LLM responses.
- Implements retry mechanisms for handling transient errors.

Example:
    client = LLMClient("mistral/mistral-small-latest")
    response = client.run("What is the meaning of life?")
    print(response)
"""
import asyncio

"""
Contributors guidelines:

*   **Python Version:** Requires Python 3.10 or later. Avoid patterns common in Python 3.6-3.9.
*   **Type Hinting:** Uses modern type hinting syntax, including PEP 604 (Union Types - `X | Y`), PEP 585 (Type Hinting Generics In Standard Collections), PEP 673 (Self Type), and others. Use type hints extensively.
*   **Data Structures:** Prefer data classes (PEP 557), `typing.NamedTuple`, or standard dictionaries/lists as appropriate. Utilize type hints for these structures.
*   **Exception Handling:** Use modern exception handling with `raise ... from ...` for exception chaining; `raise... from None` when applicable.
*   **General Style:** Follow PEP 8 guidelines.
"""

"""
    TODO: Investigate why we need to import langchainer only after load_dotenv()
        Otherwise, env vars are not available.
    
    from dotenv import load_dotenv
    load_dotenv()

    from langchainer import LLMClient


"""
import json
import httpx
import logging

from pydantic import BaseModel, Field
from typing import Any, Type, TypeVar, overload, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# from langchain_core.output_parsers import PydanticOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.language_models import BaseChatModel
# from langfuse.callback import CallbackHandler
# from langchain.schema import ChatMessage

from langchainer.logger import init_logger
from langchainer.llm_initializer import init_llm

T = TypeVar('T', bound=BaseModel)


class RateLimitError(Exception):
    """Exception raised for rate limit errors."""
    pass

class LLMInvocationError(Exception):
    """Exception raised for LLM invocation errors."""
    pass


class JSONDecodeError(json.JSONDecodeError):
    """
    Exception raised for JSON decoding errors.

    This redeclaration allows simpler inclusion of this exception in
    RetryConfig's retry_exceptions without importing it from the json package.

    Note: The LLM may initially return invalid JSON, but retries can yield valid JSON.
        Using this exception in RetryConfig's retry_exceptions allows for such retries.

    Example:
        retry_config = RetryConfig(
            attempts=5, multiplier=1, min_wait=2, max_wait=10,
            retry_exceptions=[RateLimitError, JSONDecodeError]
        )
    """
    pass


class StringResponse(BaseModel):
    """A simple model to represent a raw string response from an LLM."""
    content: str = Field(..., description="The raw string content returned by the LLM")


class RetryConfig(BaseModel):
    """
    Configuration for retry attempts.

    Defaults to 5 attempts, multiplier of 1, min wait of 2 seconds, and max wait of 10 seconds.
    Retry exceptions defaults to [RateLimitError].
    """
    attempts: int = Field(5, ge=1)
    multiplier: float = Field(1, gt=0)
    min_wait: int = Field(2, ge=0)
    max_wait: int = Field(10, ge=0)
    retry_exceptions: list[type[BaseException]] = Field(..., description="List of exception types to retry on")

class LLMClient:
    """Interface for LLM interactions"""

    def __init__(self, model: str | BaseChatModel,
                 *,
                 temperature: float | None = None,
                 max_tokens: int | None = None,
                 logger: logging.Logger | None = None,
                 log_level: int = logging.INFO,
                 rate_limiter: InMemoryRateLimiter | None = None,
                 retry_config: RetryConfig | None = None,
                 requests_per_second: int | None = None):
        """
        Initialize the LLMClient.

        Args:
            model (str | BaseChatModel): The LLM model to use.
            temperature (float | None): The temperature for the LLM. Defaults to None.
            max_tokens (int | None): The maximum number of tokens for the LLM. Defaults to None.
            logger (logging.Logger | None): The logger to use. When None, a default logger will be created.
                Defaults to None.
            log_level (int): The log level. Defaults to logging.INFO.
            rate_limiter (InMemoryRateLimiter | None): The rate limiter to use. Defaults to None.

            retry_config (RetryConfig | None): Configuration for retry attempts. Defaults to RetryConfig(attempts=5, multiplier=1, min_wait=2, max_wait=10).
                Example:
                    retry_config=RetryConfig(attempts=3, multiplier=0.5, min_wait=1, max_wait=5)

                    # Effectively disables retry by making 0 attempts
                    retry_config=RetryConfig(attempts=0)

            requests_per_second (int | None): The number of requests per second. Defaults to None.
            # Note: Providing a non-None `requests_per_second` here will eventually create an `InMemoryRateLimiter`
            # with this setting in the `init_llm` call.
        """
        self.logger = logger or init_logger(log_level)
        self.logger.info(f"Initializing LLM with model: {model}")

        self.default_retry_config = retry_config or RetryConfig(
            attempts=5, multiplier=1, min_wait=2, max_wait=10,
            retry_exceptions=[RateLimitError]
        )

        self.llm = init_llm(
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            rate_limiter=rate_limiter,
            requests_per_second=requests_per_second
        )

        # Create an event loop specifically for the synchronous `run` method.
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def _create_retry_decorator(self, config: RetryConfig = None):
        """Create a retry decorator based on the provided configuration."""
        if config is None:
            config = self.default_retry_config

        def before_sleep(retry_state):
            """Callback to log retry attempts."""
            wait_time = retry_state.next_action.sleep
            attempts_left = config.attempts - retry_state.attempt_number
            self.logger.warning(f"Retrying in {wait_time:.2f} seconds. Attempts left: {attempts_left}")

        return retry(
            stop=stop_after_attempt(config.attempts),
            wait=wait_exponential(multiplier=config.multiplier, min=config.min_wait, max=config.max_wait),
            retry=retry_if_exception_type(*config.retry_exceptions),
            before_sleep=before_sleep,
        )

    @staticmethod
    def _prepare_messages(
            prompt: str | ChatPromptTemplate | PromptTemplate,
            values: dict[str, Any],
            system_message: str,
            system_values: dict[str, Any],
            apply_xml_tags: bool | Literal['auto']
    ) -> list[BaseMessage]:
        """
        Prepare messages for LLM invocation, handling BaseModel serialization.

        Raises:
            ValueError: If `system_message` or `system_values` is provided with `ChatPromptTemplate` or `PromptTemplate`.
        """

        if isinstance(prompt, (ChatPromptTemplate, PromptTemplate)):
            if system_message is not None:
                raise ValueError("system_message should not be provided when using ChatPromptTemplate or PromptTemplate")
            if system_values:
                raise ValueError("system_values should not be provided when using ChatPromptTemplate or PromptTemplate")

        # Serialize BaseModel instances in values
        values = LLMClient._serialize_values(values)
        system_values = LLMClient._serialize_values(system_values)

        if apply_xml_tags is True or (
                apply_xml_tags == "auto"
                and _should_apply_xml_tags(prompt, values, system_message, system_values)
        ):
            if apply_xml_tags == "auto":
                print("Warning: apply_xml_tags='auto' is experimental and may not work as expected.")
            values.update({key: f"<{key}>{value}</{key}>" for key, value in values.items()})
            system_values.update({key: f"<{key}>{value}</{key}>" for key, value in system_values.items()})

        messages: list[BaseMessage] = []
        if isinstance(prompt, ChatPromptTemplate):
            messages = prompt.format_messages(**values, **system_values)
        elif isinstance(prompt, PromptTemplate):
            messages.append(HumanMessage(content=prompt.format(**values)))
        else:  # Handle string prompt
            if system_message:
                messages.append(SystemMessage(content=system_message.format(**system_values)))
            messages.append(HumanMessage(content=prompt.format(**values)))
        return messages

    @staticmethod
    def _serialize_values(values: dict[str, Any]) -> dict[str, Any]:
        """
        Serializes BaseModel objects within the template arguments recursively.
        """
        serialized_args = {}
        for key, value in values.items():
            if isinstance(value, BaseModel):
                # Default to model_dump() for LLM input
                serialized_args[key] = value.model_dump()
            elif isinstance(value, dict):
                # Recursively serialize nested dictionaries
                serialized_args[key] = LLMClient._serialize_values(value)
            elif isinstance(value, list):
                # Handle lists of BaseModel objects or other serializable types
                serialized_args[key] = [
                    item.model_dump() if isinstance(item, BaseModel) else item for item in value
                ]
            else:
                serialized_args[key] = value
        return serialized_args

    async def _invoke_llm(self, messages: list[BaseMessage], output_schema: Type[T] | Type[str]) -> T | str:
        """
        Invoke the LLM with prepared messages and parse the response.

        Args:
            messages: List of messages to send to the LLM.
            output_schema: Schema to parse the response into.

        Returns:
            Parsed response according to the output schema.

        Raises:
            RateLimitError: If the rate limit is exceeded.
            JSONDecodeError: If the response cannot be parsed as JSON.
            LLMInvocationError: If the LLM invocation fails.
        """
        try:
            if output_schema is not str:

                model_with_structure = self.llm.with_structured_output(output_schema)
                response = await model_with_structure.ainvoke(messages)
                # self.logger.debug(f"RAW LLM response: {response}", extra={'log_event': 'raw'})

                if self.logger.isEnabledFor(logging.DEBUG):
                    if isinstance(response, BaseModel):
                        self.logger.debug(response.model_dump_json(),
                                          extra={
                                              'log_event'    : 'structured_response',
                                              'output_schema': output_schema,
                                          })
                    else:
                        # Fallback to unstructured output
                        # TODO: Collect cases when structured_output response is not a BaseModel
                        #  to decide on a proper output formatting.
                        self.logger.debug(response, extra={'log_event': 'unstructured_response'})
            else:
                response = await self.llm.ainvoke(messages)
                # self.logger.debug(f"RAW LLM response: {response}", extra={'log_event': 'raw'})

                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(response, extra={'log_event': 'response'})

                if isinstance(response, dict) and 'content' in response:
                    response = response['content']
                elif isinstance(response, BaseModel) and hasattr(response, 'content'):
                    response = response.content

            return response

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                self.logger.warning("Rate limit exceeded. Retrying...")
                raise RateLimitError("Rate limit exceeded") from e
            else:
                self.logger.error(f"HTTP error in LLMClient.run: {e.response.status_code}", exc_info=True)
                raise LLMInvocationError("HTTP error during LLM invocation") from e
        except json.JSONDecodeError as e:
            self.logger.error(f"Error in parsing JSON response from LLM; for input: {messages}", exc_info=True)
            raise JSONDecodeError("Error in parsing JSON response from LLM", doc=e.doc, pos=e.pos) from e
        except Exception as e:
            self.logger.error(f"Unexpected error in LLMClient.run: {str(e)}", exc_info=True)
            self.logger.error(f"Messages at the time of error: {messages}")
            self.logger.error(f"Output schema at the time of error: {output_schema}")
            raise LLMInvocationError("Unexpected error during LLM invocation") from e

    """
        These 3 overloads enforce the following rules:
        - ChatPromptTemplate: system_message and system_values MUST NOT be provided.
        - PromptTemplate: system_message and system_values MUST NOT be provided.
        - str: system_message and system_values CAN be provided.
    """
    @overload
    async def arun(
            self,
            prompt: ChatPromptTemplate,
            values: dict[str, Any] = None,
            output_schema: Type[T] | Type[str] = str,
            apply_xml_tags: bool | Literal['auto'] = False,
            retry_config: RetryConfig | None = None,
    ) -> T | str:
        ...

    @overload
    async def arun(
            self,
            prompt: PromptTemplate,
            values: dict[str, Any] = None,
            output_schema: Type[T] | Type[str] = str,
            apply_xml_tags: bool | Literal['auto'] = False,
            retry_config: RetryConfig | None = None,
    ) -> T | str:
        ...

    @overload
    async def arun(
            self,
            prompt: str,
            values: dict[str, Any] = None,
            output_schema: Type[T] | Type[str] = str,
            system_message: str | None = None,
            system_values: dict[str, Any] = None,
            apply_xml_tags: bool | Literal['auto'] = False,
            retry_config: RetryConfig | None = None,
    ) -> T | str:
        ...
    """
        These 2 overloads handle the output schema for the LLM response:

        1. No schema specified: The output is treated as a string.
        2. Schema provided: The output is parsed according to the schema.

        Examples:
            1. Basic usage with a string output schema:
                >>> client = LLMClient("mistral/mistral-small-latest")
                >>> response = client.run("What is the meaning of life?")
                >>> print(response)
                42

            2. Basic usage with a schema:
                >>> from pydantic import BaseModel
                >>> class Entity(BaseModel):
                ...     name: str
                ...     description: str
                >>> client = LLMClient("mistral/mistral-small-latest")
                >>> response = client.run("What is the meaning of life?", output_schema=Entity)
                >>> print(response)
                Entity(name='Answer', description='42')
                
            3. Basic usage with values and output_schema:
                >>> response = await client.arun("What is the meaning of {term}?", {'term': 'life'}, Entity)
    """
    @overload
    async def arun(self,
                   prompt: str,
                   values: dict[str, Any],
                   output_schema: Type[T],
                   system_message: str = None,
                   system_values: dict[str, Any] = None
                   ) -> T:
        ...

    @overload
    async def arun(self,
                   prompt: str,
                   values: dict[str, Any] = None,
                   output_schema: Type[str] = str,
                   system_message: str = None,
                   system_values: dict[str, Any] = None
                   ) -> str:
        ...

    async def arun(self,
                   prompt: str | ChatPromptTemplate | PromptTemplate,
                   values: dict[str, Any] | None = None,
                   output_schema: Type[T] | Type[str] = str,
                   system_message: str | None = None,
                   system_values: dict[str, Any] | None = None,
                   apply_xml_tags: bool | Literal['auto'] = False,
                   retry_config: RetryConfig | None = None,
                   ) -> T | str:
        """
        Asynchronously invoke the LLM with a prompt and parse the response.

        Args:
            prompt: The prompt to send to the LLM. Can be a string, ChatPromptTemplate, or PromptTemplate.
            values: Dictionary of values to format the prompt. Defaults to None.
            output_schema: Schema to parse the response into. Defaults to `str` for unstructured responses.
            system_message: Optional system message to include. Must not be provided if `prompt` is a
                           ChatPromptTemplate or PromptTemplate. Defaults to None.
            system_values: Optional dictionary of values to format the system message. Must not be provided
                          if `prompt` is a ChatPromptTemplate or PromptTemplate. Defaults to None.
            apply_xml_tags: Whether to apply XML tags to the prompt and values. Can be `True`, `False`, or
                           `'auto'` (experimental). Defaults to False.
            retry_config: Configuration for retry attempts. Defaults to the client's default retry config.

        Returns:
            Parsed response according to the `output_schema`. If `output_schema` is `str`, returns the raw
            string response.

        Raises:
            ValueError: If `system_message` or `system_values` is provided with `ChatPromptTemplate` or
                       `PromptTemplate`.
            RateLimitError: If the rate limit is exceeded.
            JSONDecodeError: If the response cannot be parsed as JSON.
            LLMInvocationError: If the LLM invocation fails.

        Examples (illustrative):
            Basic usage with a string prompt:
            response = client.run("What is the capital of France?")
            print(response) # Paris

            Usage with a `ChatPromptTemplate` and values:
            import asyncio
            from langchain_core.prompts import ChatPromptTemplate
            async def test():
            ...     template = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
            ...     response = await client.arun(template, {"topic": "programming"})
            ...     print(response) # Or assert something about the response
            asyncio.run(test())

            Usage with a structured output schema:
            from pydantic import BaseModel
            class Joke(BaseModel):
            ...     setup: str
            ...     punchline: str
            response = client.run("Tell me a joke", output_schema=Joke)
            print(response.setup)
            print(response.punchline)

            Usage with a system message and XML tags:
            response = client.run(
            ...     "Translate '{text}' to {language}",
            ...     {"text": "Hello, world!", "language": "Spanish"},
            ...     system_message="You are a helpful translation assistant.",
            ...     apply_xml_tags=True
            ... )

            Usage with a custom retry configuration:
            _retry_config = RetryConfig(attempts=3, min_wait=1, max_wait=3, retry_exceptions=[RateLimitError])
            response = client.run("What is the meaning of life?", retry_config=_retry_config)
        """

        values = values or {}
        system_values = system_values or {}

        _retry_decorator = self._create_retry_decorator(retry_config)

        @_retry_decorator
        async def _arun_with_retry():
            """ Internal retry wrapper for arun """
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    {'system_message': system_message, 'prompt': prompt},
                    extra={
                        'log_event'    : 'prompt_template',
                        'prompt_values': {**values, **system_values},
                        'output_schema': output_schema
                    }
                )

            # if 1 is not 2:
            #     return "__DEBUG_MODE_MESSAGE__"

            messages = self._prepare_messages(prompt, values, system_message, system_values, apply_xml_tags)

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(messages, extra={'log_event': 'prompt', 'output_schema': output_schema})

            return await self._invoke_llm(messages, output_schema)

        return await _arun_with_retry()

    def run(self,
            prompt: str | ChatPromptTemplate | PromptTemplate,
            values: dict[str, Any] | None = None,
            output_schema: Type[T] | Type[str] = str,
            system_message: str | None = None,
            system_values: dict[str, Any] | None = None,
            apply_xml_tags: bool | Literal['auto'] = False,
            retry_config: RetryConfig | None = None
            ) -> T | str:
        """
        Synchronous wrapper for `arun`.

        Note: Usage examples can be found in the `arun` method docstring.
        """
        try:
            result = self._loop.run_until_complete(self.arun(
                prompt=prompt,
                values=values,
                output_schema=output_schema,
                system_message=system_message,
                system_values=system_values,
                apply_xml_tags=apply_xml_tags,
                retry_config=retry_config
            ))
            return result
        except Exception as e:
            self.logger.error(f"Error in run: {e}", exc_info=True)
            raise

    # def _handle_exception(self, e):
    #     if not isinstance(e, (RateLimitError)):
    #         self.logger.error(f"An error occurred: {e}", exc_info=True)
    #         raise

    # def invoke(self, *args, **kwargs) -> T | str:
    #     """Synchronous wrapper for arun, kept for backward compatibility."""
    #     return self.run(*args, **kwargs)
    #
    # async def ainvoke(self, *args, **kwargs) -> T | str:
    #     """Asynchronous method for arun, kept for backward compatibility."""
    #     return await self.arun(*args, **kwargs)

    def close(self):
        """Closes the event loop."""
        if self._loop:
            self._loop.close()

    def __del__(self):
        self.close()



def _should_apply_xml_tags(prompt, values, system_message, system_values):
    """
    Determine whether to apply XML tags based on the prompt and content of the values.

    !!! Experimental (uses quite dummy conditions) and may not work as expected !!!
    """
    # TODO: Improve this function to be more accurate and robust
    max_length_threshold = 100

    length_of_prompt_and_values = len(prompt) + sum(len(v) for v in values.values()) + len(system_message) + sum(
        len(v) for v in system_values.values())
    if length_of_prompt_and_values < max_length_threshold:
        return False

    has_multiple_lines = any(isinstance(v, str) and "\n" in v for v in values.values() + system_values.values())
    return has_multiple_lines
