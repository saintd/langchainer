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
import re
import json
import logging
from datetime import datetime
from typing import Any, TypedDict, Literal
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import (
    SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate,
    PromptTemplate, ChatPromptTemplate
)
from pygments.styles import get_all_styles
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.console import Console
from rich.syntax import Syntax
from rich.theme import Theme
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

DEFAULT_SYNTAX_THEME = "github-dark"
DEFAULT_PANEL_PADDING = 0


class ConfigSettings(BaseSettings):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field("INFO", alias="LOG_LEVEL")
    syntax_theme: str = Field(DEFAULT_SYNTAX_THEME, alias="SYNTAX_THEME")
    markdown_prompt: bool = Field(True, alias="MARKDOWN_PROMPT")
    markdown_response: bool = Field(True, alias="MARKDOWN_RESPONSE")
    show_prompt_values: bool = Field(True, alias="SHOW_PROMPT_VALUES")
    panel_padding: int | tuple[int, ...] = Field(0, alias="PANEL_PADDING")

    @field_validator("syntax_theme", mode="before")
    def validate_syntax_theme(cls, value):
        value = value.lower()
        if value not in get_all_styles():
            logging.warning(f"Invalid SYNTAX_THEME env var '{value}'. Using default '{DEFAULT_SYNTAX_THEME}'.")
            return DEFAULT_SYNTAX_THEME
        return value

    @field_validator("panel_padding", mode="before")
    def validate_panel_padding(cls, value):
        if isinstance(value, str):
            try:  # First, try to convert it to a single integer
                return int(value)
            except ValueError:  # If that fails, try to convert it to a tuple of integers
                try:
                    padding_tuple = tuple(map(int, value.strip("()").split(",")))
                    if not (1 <= len(padding_tuple) <= 4):
                        raise ValueError("Padding tuple must have between 1 and 4 elements")
                    return padding_tuple
                except ValueError:
                    logging.warning(f"Invalid PANEL_PADDING env var '{value}'. Using default '{DEFAULT_PANEL_PADDING}'.")
                    return DEFAULT_PANEL_PADDING
        return value

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)


config = ConfigSettings()

LOG_LEVEL = config.log_level
SYNTAX_THEME = config.syntax_theme
PANEL_PADDING = config.panel_padding
MARKDOWN_PROMPT = config.markdown_prompt
MARKDOWN_RESPONSE = config.markdown_response
SHOW_PROMPT_VALUES = config.show_prompt_values


class RichLogHandler:
    """Handles logging events and formats output using Rich."""

    def __init__(self, record: logging.LogRecord, console: Console):
        self.record = record
        self.console = console
        self.current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        self.log_event = getattr(record, 'log_event', None)
        self.prompt_values = getattr(record, 'prompt_values', None)
        self.output_schema = getattr(record, 'output_schema', None)

        self.message_types = {
            HumanMessage : ("👤 Human", "green"),
            SystemMessage: ("⚙ System", "blue"),
            AIMessage    : ("🤖 AI", "magenta")
        }
        self.prompt_template_types = {
            HumanMessagePromptTemplate : ("👤 Human Template", "green"),
            SystemMessagePromptTemplate: ("⚙ System Template", "blue"),
            AIMessagePromptTemplate    : ("🤖 AI Template", "magenta")
        }
        self.default_message_type = ("✏️ Message", "cyan")

    def handle_log_event(self) -> bool:
        """Main handler function to process log records."""

        match self.log_event:
            case "prompt_template":
                self._handle_prompt_template()
            case "prompt":
                self._handle_prompt()
            case "structured_response":
                self._handle_structured_response()
            case "response" | "unstructured_response":
                self._handle_response()
            case _:
                return False

        return True

    @staticmethod
    def _highlight_template_variables(text: str) -> Text:
        """Highlights variables (placeholders) within a template string."""
        var = Text()
        for part in re.split(r"(\{.*?})", text):
            if part.startswith("{") and part.endswith("}"):
                var.append("{", style="bold magenta")
                var.append(part[1:-1], style="bold green")
                var.append("}", style="bold magenta")
            else:
                var.append(part)
        return var

    @staticmethod
    def _extract_token_usage_and_model(llm_response: Any) -> tuple[dict, str] | None:
        """Extracts token usage information from a LangChain LLM response."""
        if isinstance(llm_response, AIMessage):
            if hasattr(llm_response, 'response_metadata'):
                metadata = llm_response.response_metadata
                return metadata.get('token_usage', {}), metadata.get('model', 'N/A')
            elif hasattr(llm_response, 'usage_metadata'):
                # Fallback; response_metadata seems to be a provider-specific
                usage = llm_response.usage_metadata
                return {
                    "prompt_tokens"    : usage.get("input_tokens"),
                    "completion_tokens": usage.get("output_tokens"),
                    "total_tokens"     : usage.get("total_tokens"),
                }, 'N/A'

        return None

    def _format_token_usage_title(self, llm_response: Any) -> str:
        """Formats the token usage information into a string."""
        token_usage, model_name = self._extract_token_usage_and_model(llm_response)
        if token_usage is None:
            return f"Model: [bold]N/A[/bold]. Token usage I/O: [bold]N/A[/bold]"
        return (f"Model: [bold]{model_name}[/bold]. Token usage I/O: {token_usage.get('prompt_tokens', 'N/A')}"
                f" + {token_usage.get('completion_tokens', 'N/A')}"
                f" = [bold]{token_usage.get('total_tokens', 'N/A')}[/bold]")

    #
    #   Handlers for specific log events
    #

    def _handle_prompt_template(self):
        """Handles and displays a prompt template."""

        class PromptComponents(TypedDict):
            system_message: str
            prompt: str | ChatPromptTemplate | PromptTemplate

        log_data: PromptComponents = self.record.msg
        messages = []

        if log_data['system_message']:
            messages.append(SystemMessagePromptTemplate.from_template(log_data['system_message']))

        if isinstance(log_data['prompt'], ChatPromptTemplate):
            messages.extend(log_data['prompt'].messages)
        elif isinstance(log_data['prompt'], PromptTemplate):
            messages.append(HumanMessagePromptTemplate.from_template(log_data['prompt'].template))
        else:
            messages.append(HumanMessagePromptTemplate.from_template(log_data['prompt']))

        grid = Table.grid(expand=True)
        grid.add_column(justify="left")

        # Display unformatted messages
        if len(messages) == 1:
            message = messages[0]
            response = self._highlight_template_variables(message.prompt.template)
            grid.add_row(response)
        else:
            for message in messages:
                message_type = type(message)
                title, style = self.prompt_template_types.get(message_type, self.default_message_type)

                if message_type in (HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate):
                    response = self._highlight_template_variables(message.prompt.template)
                else:
                    response = Text(str(message))

                grid.add_row(Panel(response, title=title, border_style=style, title_align="left", padding=PANEL_PADDING))

        # Add prompt values to the output
        if SHOW_PROMPT_VALUES and self.prompt_values:
            grid.add_row("")
            try:
                json_str = json.dumps(self.prompt_values, indent=2, default=str)
            except TypeError as e:
                error_message = f"RichLogHandler: Error serializing prompt values: {str(e)}"
                self.console.print(f"[bold red]{error_message}[/bold red]")
                self.console.print(f"[bold]Raw Prompt Values:[/bold] {self.prompt_values}")
                json_str = json.dumps({"error": error_message}, indent=2)

            prompt_values_syntax = Syntax(json_str, "json", theme=SYNTAX_THEME,
                                          line_numbers=False, word_wrap=True, padding=PANEL_PADDING)
            grid.add_row(prompt_values_syntax)

        self.console.print("")
        self.console.print(Panel(
            grid,
            title=f"📝 Prompt Template - {self.current_time}", border_style="yellow", title_align="center", padding=(
            1, 1),
            subtitle=f"Prompt type: [bold]{type(log_data['prompt']).__name__}[/bold]", subtitle_align="right"
        ))
        self.console.print("")

    def _handle_prompt(self):
        """Handles and displays a prompt."""
        messages: list[Any] = self.record.msg

        if len(messages) == 1:
            msg = messages[0]
            title, style = self.message_types.get(type(msg), self.default_message_type)
            title = f"{title} - {self.current_time}"
            response = Markdown(msg.content) if MARKDOWN_PROMPT else Text(msg.content)

            self.console.print(Panel(response, title=title, border_style=style, title_align="left", padding=PANEL_PADDING))
        else:
            grid = Table.grid(expand=True)
            grid.add_column(justify="left")

            for msg in messages:
                title, style = self.message_types.get(type(msg), self.default_message_type)
                response = Markdown(msg.content) if MARKDOWN_PROMPT else Text(msg.content)

                grid.add_row(Panel(response, title=title, border_style=style, title_align="left", padding=PANEL_PADDING))

            self.console.print(Panel(
                grid,
                title=f"💬 Prompt (Chat) - {self.current_time}", title_align="left", border_style="cyan",
                subtitle=f"Type: [bold]{type(messages).__name__}[/bold]", subtitle_align="right"
            ))

    def _handle_structured_response(self):
        """Handles and displays a structured response."""
        response = self.record.msg
        try:
            data = json.loads(response)
            json_str = json.dumps(data, indent=2)

            syntax = Syntax(json_str, lexer="json", theme=SYNTAX_THEME, line_numbers=False, word_wrap=True)
            panel = Panel(
                syntax,
                title=f"🗂  LLM Response (Structured) - {self.current_time}", title_align="left",
                border_style="green bold", padding=PANEL_PADDING,
                subtitle=f"Output schema: [bold]{self.output_schema.__name__}[/bold]", subtitle_align="right",
                expand=True
            )
            self.console.print(panel)
        except json.JSONDecodeError:
            error_message = f"JSON Parsing Error - {self.current_time}\n{response}"
            error_panel = Panel(error_message, title="Error", border_style="red", padding=PANEL_PADDING)
            self.console.print(error_panel)

    def _handle_response(self):
        """Handles and displays an unstructured response."""
        response = self.record.msg
        response_summary = self._format_token_usage_title(response)

        if isinstance(response, dict) and 'content' in response:
            response = response['content']
        elif isinstance(response, BaseModel) and hasattr(response, 'content'):
            response = response.content
        elif isinstance(response, str):
            pass
        else:
            response = str(response)

        message = Markdown(response) if MARKDOWN_RESPONSE else response

        panel = Panel(
            message,
            title=f"🤖 LLM Response - {self.current_time}", title_align="left",
            border_style="green",
            subtitle=response_summary, subtitle_align="right",
            padding=PANEL_PADDING
        )
        self.console.print(panel)


class DebugRichHandler(RichHandler):
    """Custom RichHandler to integrate with the RichLogHandler class."""

    def __init__(self, *args, **kwargs):
        custom_theme = Theme({
            "logging.level.debug"   : "violet",
            "logging.level.info"    : "blue",
            "logging.level.warning" : "yellow",
            "logging.level.error"   : "red",
            "logging.level.critical": "bold red",
        })
        self.console = Console(theme=custom_theme)
        super().__init__(
            *args,
            console=self.console,
            rich_tracebacks=True,
            omit_repeated_times=False,
            **kwargs
        )

    def emit(self, record):
        """Handles the emission of log records."""
        try:
            handler = RichLogHandler(record, self.console)
            if not handler.handle_log_event():
                super().emit(record)
        except Exception as e:
            self.console.print(f"[bold red]Error during logging:[/bold red] {str(e)}", style="red")
            self.handleError(record)
