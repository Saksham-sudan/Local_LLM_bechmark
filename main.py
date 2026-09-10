import csv 
from ollama import chat
from pydantic import BaseModel

class schema_structure(BaseModel):
  thought_process: str
  answer : str

def benchmark(model_name, prompt_set):
   with open(prompt_set, newline='') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:

        response = chat(
           model=model_name,
           messages=[{'role': 'user', 'content': row['prompt']}],
           stream=False,
           format= schema_structure.model_json_schema(),
           options= {'temperature': 0},
           )

        response_msg = schema_structure.model_validate_json(response.message.content)
        tttft_ms = response.prompt_eval_duration/1e6
        dec_token_count = response.eval_count
        dec_time_taken_sec = response.eval_duration/1e9
        tps_sec = dec_token_count/dec_time_taken_sec
        total_response_latency_sec = response.total_duration/1e9

        print("------------------------------------------------")
        print(row['prompt'])
        print(f"Thought Process: {response_msg.thought_process}")
        print(f"answer: {response_msg.answer}")
        print(f"TTTFT in ms: {tttft_ms}")
        print(f"TPS in /sec: {tps_sec}")
        print(f"Total Response Latency: {total_response_latency_sec}")

if __name__ == "__main__":
   benchmark("llama3.2:3b", "standard_prompt_set.csv")