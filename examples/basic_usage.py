import asyncio
import logging

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from langchainer import LLMClient, init_logger

async def main():

    # custom_retry = RetryConfig(attempts=3, max_wait=5)
    # client = LLMClient(retry_config=custom_retry)

    # client.run("some prompt", retry_config=RetryConfig(attempts=10, multiplier=2)) # Overrides default for this call
    # client.run("some prompt", retry_config=RetryConfig(attempts=0))  # effectively disables retry by making 0 attempts

    client = LLMClient("mistral/mistral-small-latest", log_level=logging.DEBUG)

    # result = await client.arun("Hi")

    system_message = "You are a knowledgeable and concise assistant well-versed in historical advancements."
    prompt = "What are some notable historical events in the {event}?"
    event = "AI development"
    result = await client.arun(prompt, {"event": event}, system_message=system_message)

    # print(f"Model: {args.model}\nResponse:\n", result, "\n")

    class EventAnalysis(BaseModel):
        recent_developments: list[str] = Field(..., description="List of recent developments")
        key_players: list[str] = Field(..., description="Important companies, individuals, or organizations")
        relevance_score: float = Field(..., ge=0, le=1, description="Overall relevance score (0.0-1.0)")

    result = await client.arun(prompt, {"event": event}, EventAnalysis)

    # print(f"Model: {args.model}")
    # print(f"Response:")
    # print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
