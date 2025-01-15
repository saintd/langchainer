import asyncio
import os

from models import TaskPlan, TaskEvaluation, prompt, evaluation_prompt
from dotenv import load_dotenv
import logging
from langchainer import LLMClient

load_dotenv()

# os.environ["RICH_LOGGING_ENABLED"] = "true"
# os.environ["LOG_LEVEL"] = "DEBUG"

client = LLMClient("mistral-small-latest")

task_plan = client.run_sync(prompt, {"topic": "AI ethics"}, TaskPlan)
task_evaluation = client.run_sync(evaluation_prompt, {"task_plan": task_plan}, TaskEvaluation)

# async def main():
#     task_plan = await client.run(prompt, {"topic": "AI ethics"}, TaskPlan)
#     task_evaluation = await client.run(evaluation_prompt, {"task_plan": task_plan}, TaskEvaluation)
#
# asyncio.run(main())