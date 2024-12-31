from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator


class Task(BaseModel):
    """A task with a unique id, a description, and a list of subtask ids."""
    id: int = Field(default=0, description="Unique id of the task")
    task: str = Field(..., description="The task in text form.")
    subtasks: list[int] = Field(..., description="IDs of subtasks that must be completed before the main task.")

class TaskPlan(BaseModel):
    """A task plan consisting of a list of tasks and their subtasks."""
    task_graph: list[Task] = Field(..., description="List of tasks and subtasks needed to complete the main task.")

    @field_validator("task_graph")
    def validate_task_graph(cls, task_graph):
        task_ids = [task.id for task in task_graph]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Duplicate task IDs found in task_graph")
        for task in task_graph:
            for subtask_id in task.subtasks:
                if subtask_id not in task_ids:
                    raise ValueError(f"Task {task.id} refers to non-existent subtask {subtask_id}")
        return task_graph

class TaskEvaluation(BaseModel):
    """An evaluation of a task's feasibility and completeness."""
    task_id: int = Field(..., description="ID of the task being evaluated.")
    evaluation: str = Field(..., description="Evaluation of the task's feasibility and completeness.")

# Define the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world-class task planning algorithm."),
    ("user", "I want to write a blog post about {topic}.")
])

evaluation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a task evaluation algorithm."),
    ("user", "Evaluate the feasibility of this task plan: {task_plan}")
])

