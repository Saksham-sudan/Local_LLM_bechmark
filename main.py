import time
import threading
import pandas as pd
import seaborn as sns
from pynvml import *
from ollama import chat
from pydantic import BaseModel, ValidationError, Field

class schema_structure(BaseModel):
  thought_process: str = Field(
     description = "Your step-by-step reasoning. Keep it under 15 sentences. DO NOT write the final answer here."
  )
  answer : str = Field(
     description = "The final answer ONLY. You MUST provide a value here. "
    )

shared_data = {"vram_used": 0}
data_lock = threading.Lock()
stop_event = threading.Event()

result = []

def vram_lookup():
   while not stop_event.is_set() :
      handle = nvmlDeviceGetHandleByIndex(0)
      ram_used = nvmlDeviceGetMemoryInfo(handle).used
      with data_lock:
         if ram_used > shared_data["vram_used"]:
            shared_data["vram_used"] = ram_used
      time.sleep(0.05)

def ans_qual_eval(model_ans, model_thought, expected_ans,):
   s_model_ans = model_ans.strip().lower()
   s_expected_ans = expected_ans.strip().lower()
   s_model_thought = model_thought.strip().lower()
   if s_expected_ans in s_model_ans or s_expected_ans in s_model_thought:
      return True
   else:
      return False
      
def benchmark(model_name, prompt_set):
   prompt_data = pd.read_csv(prompt_set, dtype=str)
   for row in prompt_data.itertuples(index=True):

      with data_lock:
         shared_data['vram_used'] = 0

      print("------------------------------------------------")
      print(f"Processing {row.question_id}: {row.category}...")
      response = chat(
         model=model_name,
         messages=[{"role": "system", "content": "You are a strict reasoning engine. You must separate your reasoning from your final answer for reasoning you have thought process field. The 'answer' field must contain ONLY the final target value and the answer field must never be empty. Never include explanations in the 'answer' field."}, {"role": "user", "content": row.prompt}],
         stream=False,
         format= schema_structure.model_json_schema(),
         options= {'temperature': 0, 'num_predict': 1024},
         )

      if row.category == "warmup":
         continue

      try:
         response_msg = schema_structure.model_validate_json(response.message.content)
      except ValidationError as e:
         print(f"Prompt failed: {e}")
         continue

      ttft_ms = response.prompt_eval_duration/1e6
      dec_token_count = response.eval_count
      dec_time_taken_sec = response.eval_duration/1e9
      tps_sec = dec_token_count/dec_time_taken_sec if dec_time_taken_sec > 0 else 0.0
      total_response_latency_sec = response.total_duration/1e9
      category = row.category
      is_correct = ans_qual_eval(response_msg.answer, response_msg.thought_process, row.expected_output)

      with data_lock:
         vram_usuage_mb = shared_data['vram_used'] / (1024**2)

      print(row.prompt)
      print(f"Thought Process: {response_msg.thought_process}")
      print(f"answer: {response_msg.answer}")
      print(f"passed: {is_correct}")
      print(f"TTFT in Ms: {ttft_ms}")
      print(f"TPS in /sec: {tps_sec}")
      print(f"Total Response Latency in Sec: {total_response_latency_sec}")
      print(f"Vram Usuage in Mb: {vram_usuage_mb}")

      result.append({
         "quesstion_id": row.question_id,
         "category": row.category,
         "passed": is_correct
      })

      data_out = pd.DataFrame(result)
      data_out.to_csv("test.csv", index = False)

if __name__ == "__main__":
   nvmlInit()
   monitor_thread = threading.Thread(target=vram_lookup, daemon=True)
   monitor_thread.start()

   try:
      benchmark("llama3.2:3b", "prompt_set.csv")
   finally:
      stop_event.set()
      monitor_thread.join()
      nvmlShutdown()