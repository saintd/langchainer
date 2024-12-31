
from langchain_mistralai import ChatMistralAI

from models import TaskPlan, TaskEvaluation, prompt, evaluation_prompt



llm = ChatMistralAI(model="mistral-small-latest")
prompt_value = prompt.invoke({"topic": "AI ethics"})
structured_llm = llm.with_structured_output(TaskPlan)
task_plan = structured_llm.invoke(prompt_value)


evaluation_prompt_value = evaluation_prompt.invoke({"task_plan": task_plan})
structured_llm_eval = llm.with_structured_output(TaskEvaluation)
task_evaluation = structured_llm_eval.invoke(evaluation_prompt_value)

print(task_plan)
print(task_evaluation)