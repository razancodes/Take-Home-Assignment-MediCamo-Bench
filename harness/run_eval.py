import json
import os
import time
from datetime import datetime
from prompts import build_prompt
from models import MODEL_ROUTERS

GOLD_FILE = r"c:\Users\MRaza\Documents\TH-Assignment-AI-data\clinical-llm-benchmark\gold\gold_labels.json"
OUTPUT_DIR = r"c:\Users\MRaza\Documents\TH-Assignment-AI-data\clinical-llm-benchmark\harness\outputs"

def load_env():
    env_file = r"c:\Users\MRaza\Documents\TH-Assignment-AI-data\clinical-llm-benchmark\.env"
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

def main():
    load_env()
    
    with open(GOLD_FILE, "r", encoding="utf-8") as f:
        gold_items = json.load(f)
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for item in gold_items:
        item_id = item["item_id"]
        excerpt = item["excerpt"]
        question = item["question"]
        
        prompt = build_prompt(excerpt, question)
        
        for model_name, model_fn in MODEL_ROUTERS.items():
            safe_model_name = model_name.replace("/", "_").replace(":", "_")
            
            # Check if this model/item combination already has a valid output
            existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"{item_id}_{safe_model_name}")]
            if existing_files:
                print(f"Skipping {item_id} on {model_name} (already exists)")
                continue
                
            print(f"Running {item_id} on {model_name}...")
            
            try:
                # Add delay to avoid rate limiting
                time.sleep(2)
                response = model_fn(prompt)
                
                output_filename = f"{item_id}_{safe_model_name}_{timestamp}.json"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                output_data = {
                    "item_id": item_id,
                    "model_version": model_name,
                    "provider": "HuggingFace Inference API",
                    "interface": "API",
                    "prompt": prompt,
                    "response": response
                }
                
                with open(output_path, "w", encoding="utf-8") as out_f:
                    json.dump(output_data, out_f, indent=2)
                
                print(f"Saved {output_path}")
                
            except Exception as e:
                print(f"Error running {item_id} on {model_name}: {e}")

if __name__ == "__main__":
    main()
