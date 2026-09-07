import csv 
from ollama import chat
from pydantic import BaseModel

class schema_structure(BaseModel):
  thought_process: str
  answer : str

with open('test_file.csv', newline='') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        question= row['prompt']
        response = chat(
    model='llama3.2:3b',
    messages=[{'role': 'user', 'content': question}],
    stream=False,
    format= schema_structure.model_json_schema()
)

print(response)

