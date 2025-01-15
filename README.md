# LangChainer

[![PyPI version](https://badge.fury.io/py/langchainer.svg)](https://badge.fury.io/py/langchainer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


LangChainer is a lightweight Python library that streamlines your interactions with Large Language Models (LLMs) by bringing together the best of **LangChain**, **LiteLLM**, **Instructor**, and **Rich** into a single, easy-to-use package. It provides a unified interface for working with various LLM providers, simplifies structured output handling, and enhances debugging with beautiful, informative logging.
Offers incredible simplicity for LLM interaction and enhanced debugging, especially with structured output.

**Key Features:**

*   **Simplified LLM Interaction:** One helper (`init_llm`) and one client (`LLMClient`) to rule them all.
*   **Unified Interface:**  Work seamlessly with multiple LLM providers (OpenAI, Mistral, Together, Anthropic, OpenRouter, Google Gemini) using a consistent API.
*   **Structured Output:** Easily define and receive structured data from LLMs using Pydantic models.
*   **Enhanced Debugging:** Leverage the power of [Rich](https://github.com/Textualize/rich) for visually appealing and informative logs, making debugging a breeze.
*   **Rate Limiting & Retries:** Built-in rate limiting and configurable retry mechanisms for robust LLM interactions.
*   **Modern Python:** Developed for Python 3.10+ with extensive type hinting and a clean, modern codebase.

**Installation**

Install the core `langchainer` package:

```bash
pip install langchainer
```

For enhanced debugging with Rich, install the `[rich]` extra:

```bash
pip install langchainer[rich]
```

To use a specific LLM provider, install the corresponding provider package in the standard way (e.g. `pip install langchain_openai` for OpenAI) or use the shortcut extras:
```bash
pip install langchainer[openai]  # For OpenAI
pip install langchainer[mistral] # For Mistral
pip install langchainer[anthropic] # For Anthropic
...
pip install langchainer[rich,mistral] # For Mistral with Rich. And so on.
```

**Basic Usage**

```dotenv
# .env
OPENAI_API_KEY="YOUR_API_KEY"
RICH_LOGGING_ENABLED=true
```

```python
import logging
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from langchainer import LLMClient

# Optional: Enable Rich logging
# from langchainer import init_logger
# init_logger(log_level=logging.DEBUG, use_rich=True)

# Initialize the client with your chosen model
client = LLMClient("mistral/mistral-small-latest")  # Or "openai/gpt-3.5-turbo", etc.

# --- Example 1: Simple String Prompt ---
response = await client.run("What is the meaning of life?")
print(response)

# --- Example 2: Using a ChatPromptTemplate ---
template = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
response = await client.run(template, {"topic": "programming"})
print(response)


# --- Example 3: Structured Output ---
class Joke(BaseModel):
    setup: str
    punchline: str


response = client.run_sync("Tell me a joke about technology", output_schema=Joke)
print(response.setup)
print(response.punchline)

# --- Example 4: Async usage with structured output and XML tags ---
import asyncio


async def main():
    class Fact(BaseModel):
        fact: str

    response = await client.run(
        "Tell me a fact about {subject}",
        {"subject": "the ocean"},
        output_schema=Fact,
        system_message="You are a helpful assistant.",
        apply_xml_tags=True
    )
    print(response)


asyncio.run(main())
```

**Configuration**

You can configure `langchainer`'s logging behavior using environment variables:

*   **`LOG_LEVEL`:**  Set the logging level (e.g., `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). Default is `INFO`.
*   **`RICH_LOGGING_ENABLED`:** Enable or disable Rich logging (`true` or `false`). Default is `false`.
*   **`SYNTAX_THEME`:**  Set the syntax highlighting theme for Rich (see [available themes](https://pygments.org/styles/)). Default is `github-dark`.
*   **`MARKDOWN_PROMPT`:** Enable or disable Markdown rendering for prompt content (default is `true`).
*   **`MARKDOWN_RESPONSE`:** Enable or disable Markdown rendering for LLM response content (default is `true`).
*   **`SHOW_PROMPT_VALUES`:**  Enable or disable the display of prompt variable values in the log output (default is `true`).
*   **`PANEL_PADDING`:** Set the padding for Rich panels (integer or tuple, default is `0`).

**Documentation**

For more advanced usage, detailed API documentation, and examples, please visit our GitHub repository.

**Contributing**

Contributions are welcome! Please see our [Contribution Guidelines](CONTRIBUTING.md) for details.

**License**

`langchainer` is released under the MIT License. See the [LICENSE](LICENSE) file for details.

**Disclaimer**

`langchainer` is an independent project and is not officially affiliated with LangChain.

