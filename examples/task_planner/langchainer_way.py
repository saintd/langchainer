import asyncio
from models import TaskPlan, TaskEvaluation, prompt, evaluation_prompt
from dotenv import load_dotenv
import logging
load_dotenv()

from langchainer import LLMClient


# client = LLMClient("mistral/mistral-small-latest", log_level=logging.DEBUG)
client = LLMClient("mistral-small-latest", log_level=logging.DEBUG)

# task_plan = client.run(prompt, {"topic": "AI ethics"}, TaskPlan)
# task_evaluation = client.run(evaluation_prompt, {"task_plan": task_plan}, TaskEvaluation)

async def main():
    task_plan = await client.arun(prompt, {"topic": "AI ethics"}, TaskPlan)
    task_evaluation = await client.arun(evaluation_prompt, {"task_plan": task_plan}, TaskEvaluation)

asyncio.run(main())