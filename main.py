import csv 
from ollama import chat
from pydantic import BaseModel

class schema_structure(BaseModel):
  thought_process: str
  answer : str

with open('test_file.csv', newline='') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        response = chat(
            model='llama3.2:3b',
            messages=[{'role': 'user', 'content': row['prompt']}],
            stream=False,
            format= schema_structure.model_json_schema(),
            options= {'temperature': 0},
        )
        response_tlm = response.prompt_eval_count
        response_msg = schema_structure.model_validate_json(response.message.answer)
        print(response_msg)
        print(response_tlm)


