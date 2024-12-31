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
import os
import re
import json
import logging
from datetime import datetime
from pydantic import BaseModel
from typing import TypedDict, Any

# from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import (
    SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate,
    PromptTemplate, ChatPromptTemplate
)

from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.console import Console
from rich.syntax import Syntax
from rich.theme import Theme
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def load_config_from_env() -> dict:
    """Load configuration from environment variables."""
    default_syntax_theme = "github-dark"
    config_settings = {
        "LOG_LEVEL"         : os.environ.get("LOG_LEVEL", "INFO").upper(),
        "SYNTAX_THEME"      : os.environ.get("SYNTAX_THEME", default_syntax_theme).lower(),
        "MARKDOWN_PROMPT"   : os.environ.get("MARKDOWN_PROMPT", "true").lower() == "true",
        "MARKDOWN_RESPONSE" : os.environ.get("MARKDOWN_RESPONSE", "true").lower() == "true",
        "SHOW_PROMPT_VALUES": os.environ.get("SHOW_PROMPT_VALUES", "true").lower() == "true",
        "PANEL_PADDING"     : os.environ.get("PANEL_PADDING", 0),
    }

    # Padding can be an integer (e.g., 0, 1, 2) or a tuple of integers (e.g., (0, 1), (1, 1), (2, 1, 0, 1))
    padding_value = config_settings["PANEL_PADDING"]
    if isinstance(padding_value, str):
        try:
            # First, try to convert it to a single integer
            config_settings["PANEL_PADDING"] = int(padding_value)
        except ValueError:
            # If that fails, try to convert it to a tuple of integers
            try:
                config_settings["PANEL_PADDING"] = tuple(map(int, padding_value.strip('()').split(',')))  # noqa
            except ValueError:
                print(f"Warning: Invalid PANEL_PADDING env var '{padding_value}'. Using default '0'.")
                config_settings["PANEL_PADDING"] = 0

    if config_settings["SYNTAX_THEME"] != default_syntax_theme:
        from pygments.styles import get_all_styles
        if config_settings["SYNTAX_THEME"] not in get_all_styles():
            print(f"Warning: Invalid SYNTAX_THEME env var '{config_settings['SYNTAX_THEME']}'. Using default '{default_syntax_theme}'.")
            config_settings["SYNTAX_THEME"] = default_syntax_theme

    return config_settings


config = load_config_from_env()
LOG_LEVEL = config["LOG_LEVEL"]
SYNTAX_THEME = config["SYNTAX_THEME"]
PANEL_PADDING = config["PANEL_PADDING"]
MARKDOWN_PROMPT = config["MARKDOWN_PROMPT"]
MARKDOWN_RESPONSE = config["MARKDOWN_RESPONSE"]
SHOW_PROMPT_VALUES = config["SHOW_PROMPT_VALUES"]


def debug_rich_handler(record: logging.LogRecord, console: Console) -> bool:
    """Handler for Rich logging."""
    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    log_event = getattr(record, 'log_event', None)
    prompt_values = getattr(record, 'prompt_values', None)
    output_schema = getattr(record, 'output_schema', None)

    message_types = {
        HumanMessage : ("👤 Human", "green"),
        SystemMessage: ("⚙ System", "blue"),
        AIMessage    : ("🤖 AI", "magenta")
    }
    prompt_template_types = {
        HumanMessagePromptTemplate : ("👤 Human Template", "green"),
        SystemMessagePromptTemplate: ("⚙ System Template", "blue"),
        AIMessagePromptTemplate    : ("🤖 AI Template", "magenta")
    }
    default_message_type = ("✏️ Message", "cyan")

    def highlight_template_variables(text: str) -> Text:
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

    def extract_token_usage_and_model(llm_response: Any) -> tuple[dict, str] | None:
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

    def format_token_usage_title(llm_response: Any) -> str:
        token_usage, model_name = extract_token_usage_and_model(llm_response)
        if token_usage is None:
            return f"Model: [bold]N/A[/bold]. Token usage I/O: [bold]N/A[/bold]"
        return (f"Model: [bold]{model_name}[/bold]. Token usage I/O: {token_usage.get('prompt_tokens', 'N/A')}"
                f" + {token_usage.get('completion_tokens', 'N/A')}"
                f" = [bold]{token_usage.get('total_tokens', 'N/A')}[/bold]")

    #
    #   Prompt template
    #
    if log_event == "prompt_template":

        class PromptComponents(TypedDict):
            system_message: str
            prompt: str | ChatPromptTemplate | PromptTemplate

        log_data: PromptComponents = record.msg
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
            # Single message (Human or System) - no nested panel is created for it
            message = messages[0]
            response = highlight_template_variables(message.prompt.template)
            grid.add_row(response)
        else:
            # Multiple messages
            for message in messages:
                message_type = type(message)
                title, style = prompt_template_types.get(message_type, default_message_type)

                if message_type in (HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate):
                    response = highlight_template_variables(message.prompt.template)
                else:
                    response = Text(str(message))

                grid.add_row(Panel(response, title=title, border_style=style, title_align="left", padding=PANEL_PADDING))

        # Add prompt values to the output
        if SHOW_PROMPT_VALUES and prompt_values:
            grid.add_row("")
            prompt_values_syntax = Syntax(
                json.dumps(prompt_values, indent=2), "json", theme=SYNTAX_THEME, line_numbers=False, padding=PANEL_PADDING
            )
            grid.add_row(prompt_values_syntax)

        console.print("")  # Add margin
        console.print(Panel(
            grid,
            title=f"📝 Prompt Template - {current_time}", border_style="yellow", title_align="center", padding=(1, 1),
            subtitle=f"Prompt type: [bold]{type(log_data['prompt']).__name__}[/bold]", subtitle_align="right"
        ))
        console.print("")  # Add margin

    #
    #   Prompt
    #
    elif log_event == "prompt":
        messages: list[Any] = record.msg

        if len(messages) == 1:
            # Single message (Human or System)
            msg = messages[0]
            title, style = message_types.get(type(msg), default_message_type)
            title = f"{title} - {current_time}"
            response = Markdown(msg.content) if MARKDOWN_PROMPT else Text(msg.content)

            console.print(Panel(response, title=title, border_style=style, title_align="left", padding=PANEL_PADDING))
        else:
            # Multiple messages (ChatPromptTemplate with multiple messages)
            grid = Table.grid(expand=True)
            grid.add_column(justify="left")

            for msg in messages:
                title, style = message_types.get(type(msg), default_message_type)
                response = Markdown(msg.content) if MARKDOWN_PROMPT else Text(msg.content)

                grid.add_row(Panel(response, title=title, border_style=style, title_align="left", padding=PANEL_PADDING))

            console.print(Panel(
                grid,
                title=f"💬 Prompt (Chat) - {current_time}", title_align="left", border_style="cyan",
                subtitle=f"Type: [bold]{type(messages).__name__}[/bold]", subtitle_align="right"

            ))

    #
    #   Structured Response
    #
    elif log_event == "structured_response":
        response = record.msg
        try:
            data = json.loads(response)
            json_str = json.dumps(data, indent=2)

            syntax = Syntax(json_str, lexer="json", theme=SYNTAX_THEME, line_numbers=False)
            panel = Panel(
                syntax,
                title=f"🗂️ LLM Response (Structured) - {current_time}", title_align="left",
                border_style="green",
                subtitle=f"Output schema: [bold]{output_schema.__name__}[/bold]", subtitle_align="right"
            )
            console.print(panel)
        except json.JSONDecodeError:
            error_message = f"JSON Parsing Error - {current_time}\n{response}"
            error_panel = Panel(error_message, title="Error", border_style="red", padding=PANEL_PADDING)
            console.print(error_panel)

    #
    #   Unstructured Response
    #
    elif log_event in ("response", "unstructured_response"):
        response = record.msg
        response_summary = format_token_usage_title(response)

        if isinstance(response, dict) and 'content' in response:
            response = response['content']
        elif isinstance(response, BaseModel) and hasattr(response, 'content'):
            response = response.content
        elif isinstance(response, str):
            # If response is already a string, use it as is
            pass
        else:
            # If response is neither a dict, BaseModel, nor string, convert it to string
            response = str(response)

        message = Markdown(response) if MARKDOWN_RESPONSE else response

        panel = Panel(
            message,
            title=f"🤖 LLM Response - {current_time}", title_align="left",
            border_style="green",
            subtitle=response_summary, subtitle_align="right",
            padding=PANEL_PADDING
        )
        console.print(panel)
    else:
        return False

    return True


class DebugRichHandler(RichHandler):
    def __init__(self, *args, **kwargs):
        custom_theme = Theme({
            "logging.level.debug"   : "violet",
            "logging.level.info"    : "blue",
            "logging.level.warning" : "yellow",
            "logging.level.error"   : "red",
            "logging.level.critical": "bold red",
        })
        self.console = Console(theme=custom_theme)
        # TODO: Make this configurable via env variables.
        #  There are few nice features in Rich library.
        super().__init__(
            *args,
            console=self.console,
            rich_tracebacks=True,
            omit_repeated_times=False,
            **kwargs
        )

    def emit(self, record):
        try:
            if not debug_rich_handler(record, self.console):
                super().emit(record)
        except Exception as e:
            self.console.print(f"[bold red]Error during logging:[/bold red] {str(e)}", style="red")
            self.handleError(record)
