import logging
from pydantic import BaseModel, Field


from dotenv import load_dotenv
load_dotenv()

from langchainer import LLMClient


class Task(BaseModel):
    id: int = Field(..., description="Unique id of the task")
    task: str = Field(..., description="Contains the task in text form. If there are multiple tasks, this task can only be executed when all dependant subtasks have been answered.",)
    subtasks: list[int] = Field(..., description="IDs of subtasks that must be completed before the main task. Use when multiple questions are needed to resolve unknowns. Dependencies should only be other tasks.",)

class TaskPlan(BaseModel):
    task_graph: list[Task] = Field(..., description="List of tasks and subtasks that need to be done to complete the main task. Consists of the main task and its dependencies.")

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world-class task planning algorithm capable of breaking apart tasks into dependent subtasks, such that the answers can be used to enable the system completing the main task. "),
    ("user", "I want to write a blog post about {topic}.")])
topic = "AI ethics"


# # Chained way
# llm = ChatMistralAI(model="mistral-small-latest")
# chain = prompt | llm | PydanticOutputParser(pydantic_object=TaskPlan)
# result = chain.invoke({"topic": topic})
#
#
# # Standard way
# llm = ChatMistralAI(model="mistral-small-latest")
# prompt_value = prompt.invoke({"topic": topic})
# structured_llm = llm.with_structured_output(TaskPlan)
# result = structured_llm.invoke(prompt_value)


# LangChainer way
client = LLMClient("mistral/mistral-small-latest", log_level=logging.DEBUG)

result = client.run(prompt, {"topic": topic}, TaskPlan)



# llm = ChatMistralAI(model="mistral-small-latest")
# prompt_value = prompt.invoke({"topic": topic})
# structured_llm = llm.with_structured_output(TaskPlan)
# result = structured_llm.invoke(prompt_value)



# evaluation_chain = prompt | self.llm | StrOutputParser()
# evaluation = evaluation_chain.invoke({"plan": plan.model_dump_json(), "results": results})