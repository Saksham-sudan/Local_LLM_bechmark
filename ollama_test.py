from ollama import chat
from pydantic import BaseModel

class schema_structure(BaseModel):
  thought_process: str
  answer : str

response = chat(
    model='llama3.2:3b',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    stream=False,
    format= schema_structure.model_json_schema()
)

print(response)