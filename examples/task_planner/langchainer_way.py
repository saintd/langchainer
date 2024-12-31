from models import TaskPlan, TaskEvaluation, prompt, evaluation_prompt
from dotenv import load_dotenv
import logging
load_dotenv()

from langchainer import LLMClient

def main():
    client = LLMClient("mistral/mistral-small-latest", log_level=logging.DEBUG)

    task_plan = client.run(prompt, {"topic": "AI ethics"}, TaskPlan)
    task_evaluation = client.run(evaluation_prompt, {"task_plan": task_plan}, TaskEvaluation)
    task_evaluation = client.arun(evaluation_prompt, {"task_plan": task_plan}, TaskEvaluation, system_message="You are a task evaluation algorithm.")
    try:
        pass


    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()