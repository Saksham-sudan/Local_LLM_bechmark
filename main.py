import csv 
import time
import threading
from pynvml import *
from ollama import chat
from pydantic import BaseModel

class schema_structure(BaseModel):
  thought_process: str
  answer : str

shared_data = {"vram_used": 0}
data_lock = threading.Lock()
stop_event = threading.Event()

score_card = {
   "logic": {"Total": 0, "passed": 0},
   "math": {"Total": 0, "passed": 0},
   "knowledge": {"Total": 0, "passed": 0},
   "coding": {"Total": 0, "passed": 0},
   "data_extraction": {"Total": 0, "passed": 0},
}

def vram_lookup():
   while not stop_event.is_set() :
      handle = nvmlDeviceGetHandleByIndex(0)
      ram_used = nvmlDeviceGetMemoryInfo(handle).used
      with data_lock:
         if ram_used > shared_data["vram_used"]:
            shared_data["vram_used"] = ram_used
      time.sleep(0.05)

def ans_qual_eval(model_ans, expected_ans, eval_type):
   s_model_ans = model_ans.strip().lower()
   s_expected_ans = expected_ans.strip().lower()
   if eval_type == "exact":
       if s_model_ans == s_expected_ans:
            return True
       else:
            return False
   elif eval_type == "contains":
       if s_expected_ans in s_model_ans:
            return True
       else:
            return False
      
def benchmark(model_name, prompt_set):
    with open(prompt_set, newline='') as csv_file:
       csv_reader = csv.DictReader(csv_file)
       for row in csv_reader:
          with data_lock:
             shared_data['vram_used'] = 0
          response = chat(
             model=model_name,
             messages=[{"role": "system", "content": "You are a concise reasoning engine. Think step-by-step"},
                       {"role": "user", "content": row['prompt']}],
             stream=False,
             format= schema_structure.model_json_schema(),
             options= {'temperature': 0},
             )
          
          if row["category"] == "warmup":
             continue

          response_msg = schema_structure.model_validate_json(response.message.content)
          ttft_ms = response.prompt_eval_duration/1e6
          dec_token_count = response.eval_count
          dec_time_taken_sec = response.eval_duration/1e9
          tps_sec = dec_token_count/dec_time_taken_sec
          total_response_latency_sec = response.total_duration/1e9
          with data_lock:
             vram_usuage_mb = shared_data['vram_used'] / (1024**2)
          category = row["category"]

          print("------------------------------------------------")
          print(row['prompt'])
          print(f"Thought Process: {response_msg.thought_process}")
          print(f"answer: {response_msg.answer}")
          print(ans_qual_eval(response_msg.answer, row["expected_output"], row["evalution_type"]))
          print(f"TTFT in Ms: {ttft_ms}")
          print(f"TPS in /sec: {tps_sec}")
          print(f"Total Response Latency in Sec: {total_response_latency_sec}")
          print(f"Vram Usuage in Mb: {vram_usuage_mb}")

          total = score_card[category]["Total"]
          passed = score_card[category]["passed"]
          total += 1
          if ans_qual_eval(response_msg.answer, row["expected_output"], row["evalution_type"]):
            passed += 1
          print(f"quality: {(passed / total ) * 100}")


if __name__ == "__main__":
   nvmlInit()
   monitor_thread = threading.Thread(target=vram_lookup, daemon=True)
   monitor_thread.start()

   try:
      benchmark("llama3.2:3b", "standard_prompt_set.csv")
   finally:
      stop_event.set()
      monitor_thread.join()
      nvmlShutdown()