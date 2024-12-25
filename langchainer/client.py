"""
This module is written for Python 3.10+ and utilizes modern Python best practices.

Key features and guidelines:

*   **Python Version:** Requires Python 3.10 or later. Avoid patterns common in Python 3.6-3.9.
*   **Type Hinting:** Uses modern type hinting syntax, including PEP 604 (Union Types - `X | Y`), PEP 585 (Type Hinting Generics In Standard Collections), PEP 673 (Self Type), and others. Use type hints extensively.
*   **Data Structures:** Prefer data classes (PEP 557), `typing.NamedTuple`, or standard dictionaries/lists as appropriate. Utilize type hints for these structures.
*   **Exception Handling:** Use modern exception handling with `raise ... from ...` for exception chaining.
*   **General Style:** Follow PEP 8 guidelines.

This module aims to be a modern example of Python 3.10+ code. Refrain from using outdated patterns or workarounds typical of earlier Python 3 versions.
"""
import logging

from typing import Any, Type, TypeVar, overload, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field

# from langchain_core.output_parsers import PydanticOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.language_models import BaseChatModel
# from langfuse.callback import CallbackHandler
# from langchain.schema import ChatMessage

from langchainer.logger import init_logger
from langchainer.llm_initializer import init_llm

# T = TypeVar('T', bound=Union[BaseModel, str])
T = TypeVar('T', bound=BaseModel)

class StringResponse(BaseModel):
    """A simple model to represent a raw string response from an LLM."""
    content: str = Field(..., description="The raw string content returned by the LLM")

class RetryConfig(BaseModel):
    attempts: int = Field(5, ge=1)
    multiplier: float = Field(1, gt=0)
    min_wait: int = Field(2, ge=0)
    max_wait: int = Field(10, ge=0)



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
            model: The LLM model to use.
            temperature: The temperature for the LLM.
            max_tokens: The maximum number of tokens for the LLM.
            logger: The logger to use.
            log_level: The log level.
            rate_limiter: The rate limiter to use.
            retry_config: The retry configuration.
            requests_per_second: The number of requests per second.
        """
        self.logger = logger or init_logger(log_level)
        self.logger.info(f"Initializing LLM with model: {model}")

        self.default_retry_config = retry_config or RetryConfig(attempts=5, multiplier=1, min_wait=2, max_wait=10)

        self.llm = init_llm(
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            rate_limiter=rate_limiter,
            requests_per_second=requests_per_second
        )

    def _create_retry_decorator(self, config: RetryConfig = None):
        """Create a retry decorator based on the provided configuration."""
        if config is None:
            config = self.default_retry_config

        return retry(
            stop=stop_after_attempt(config.attempts),
            wait=wait_exponential(multiplier=config.multiplier, min=config.min_wait, max=config.max_wait)
        )

    @staticmethod
    def _prepare_messages(
            prompt: str | ChatPromptTemplate | PromptTemplate,
            values: dict[str, Any],
            system_message: str,
            system_values: dict[str, Any],
            apply_xml_tags: bool | Literal['auto']
    ) -> list[BaseMessage]:
        """Helper function to prepare messages for LLM invocation."""

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

    async def _invoke_llm(self, messages: list[BaseMessage], output_schema: Type[T] | Type[str]) -> T | str:
        """Helper function to invoke the LLM with prepared messages."""
        try:
            if output_schema is not str:
                model_with_structure = self.llm.with_structured_output(output_schema)
                response = await model_with_structure.ainvoke(messages)

                # TODO: [Maybe] consider to use raw LLM response, ??just to have a token_usage??
                # # Invoke the LLM without structured output first
                # raw_response = await self.llm.ainvoke(messages)
                # token_usage = raw_response.get('usage', 'N/A')
                # parsed_response = output_schema.parse_raw(raw_response['content'])

                token_usage = 'N/A'

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

                # self.logger.debug(f"RAW LLM response: {response}", extra={'log_event': 'token_usage'})
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(response, extra={'log_event': 'response'})

                if isinstance(response, dict) and 'content' in response:
                    response = response['content']
                elif isinstance(response, BaseModel) and hasattr(response, 'content'):
                    response = response.content


            return response

        except Exception as e:
            if "429" in str(e):
                # TODO: { Add an option | consider changes } to retry only on rate limit errors `if isinstance(e, ??RateLimitError??):`
                #  (and possibly some parsing errors - sometimes retries return properly formatted responses)
                #  - as it currently retries on any exception even if we receive a correct response.
                #  Collect and list raised exceptions here (for future consideration):
                #  - ...
                self.logger.warning("Rate limit exceeded. Retrying...")
                raise
            self.logger.error(f"Error in LLMClient.run: {str(e)}")
            raise


    """
        These overloads are needed to support the following use cases:
            1. The user does not specify an output schema, so the output will be considered as a str.
            2. The user specifies an output schema, so the output will be structured according to the schema.

        Examples:
            1. Basic usage with a string output schema:
                >>> client = LLMClient("mistral/mistral-small-latest")
                >>> response = client.run("What is the meaning of life?")
                >>> print(response)
                42

            2. Basic usage with a structured output schema:
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
        """Invokes the LLM and parses the response, ensuring structured output."""

        values = values or {}
        system_values = system_values or {}

        _retry_decorator = self._create_retry_decorator(retry_config)

        @_retry_decorator
        async def _arun_with_retry():

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

    def run(self, *args, **kwargs) -> T | str:
        """Synchronous wrapper for arun."""
        import asyncio
        return asyncio.run(self.arun(*args, **kwargs))

    # def invoke(self, *args, **kwargs) -> T | str:
    #     """Synchronous wrapper for arun, kept for backward compatibility."""
    #     return self.run(*args, **kwargs)
    #
    # async def ainvoke(self, *args, **kwargs) -> T | str:
    #     """Asynchronous method for arun, kept for backward compatibility."""
    #     return await self.arun(*args, **kwargs)

def _should_apply_xml_tags(prompt, values, system_message, system_values):
    """Determine whether to apply XML tags based on the prompt and content of the values."""
    # !!! Experimental (uses quite dummy conditions) and may not work as expected !!!
    # TODO: depends on the length of the values and system_values, decide whether to apply XML tags or not.
    #   Condition (to consider) to apply XML tags:
    #   - A value has a length greater than 100 characters.
    #   - The prompt has a length greater than 100 characters.
    #   - A value has multiple lines.
    #   Condition to disable XML tags:
    #   - Prompt + value has a length less than 100 characters.
    length_of_prompt_and_values = len(prompt) + sum(len(v) for v in values.values()) + len(system_message) + sum(len(v) for v in system_values.values())
    if length_of_prompt_and_values < 100:
        return False
    has_multiple_lines = any(isinstance(v, str) and "\n" in v for v in values.values() + system_values.values())
    return has_multiple_lines
