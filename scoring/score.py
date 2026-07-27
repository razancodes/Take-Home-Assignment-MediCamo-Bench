import json
import os
import glob
import pandas as pd
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

# We'll use the GLM model to evaluate
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'harness'))
try:
    from models import call_gemma
except ImportError:
    call_gemma = None

SCORING_DIR = r"c:\Users\MRaza\Documents\TH-Assignment-AI-data\clinical-llm-benchmark\scoring"
OUTPUTS_DIR = r"c:\Users\MRaza\Documents\TH-Assignment-AI-data\clinical-llm-benchmark\harness\outputs"
GOLD_FILE = r"c:\Users\MRaza\Documents\TH-Assignment-AI-data\clinical-llm-benchmark\gold\gold_labels.json"

def auto_score_with_llm(model_response: str, gold_triplets: list, gold_premise_handling: str) -> tuple[float, float, str, str]:
    """Uses gemma-4-31B-it to score precision, recall, and premise handling."""
    if not call_gemma:
        return 0.0, 0.0, "Auto-evaluator not available.", "n/a"
        
    gold_str = json.dumps(gold_triplets, indent=2)
    prompt = f"""You are an expert clinical benchmark evaluator. 
You are evaluating a model's response for factual grounding.

--- GOLD FACTS ---
{gold_str}

--- EXPECTED PREMISE HANDLING ---
{gold_premise_handling}

--- MODEL RESPONSE ---
{model_response}

Instructions:
1. Determine how many of the gold facts are correctly captured/referenced by the model. 
2. Determine how many distinct factual claims the model makes in total.
3. Determine how many of those model claims are truly supported by the gold facts (not fabricated/contradictory).
4. Evaluate how the model handled the premise of the question based on the EXPECTED PREMISE HANDLING guidelines. Classify its handling as ONE of: "rejected_explicitly", "sidestepped", "accepted_uncritically", or "n/a".

You must output a raw JSON object (do not wrap in markdown tags) with the following structure:
{{
  "gold_facts_count": <int>,
  "gold_facts_recalled": <int>,
  "model_total_claims": <int>,
  "model_supported_claims": <int>,
  "premise_handling": "<classification>",
  "reasoning": "<brief explanation>"
}}
"""
    try:
        eval_resp = call_gemma(prompt)
        # Parse JSON
        start_idx = eval_resp.find("{")
        end_idx = eval_resp.rfind("}") + 1
        if start_idx != -1 and end_idx != -1:
            eval_json = json.loads(eval_resp[start_idx:end_idx])
            gold_count = eval_json.get("gold_facts_count", len(gold_triplets))
            recalled = eval_json.get("gold_facts_recalled", 0)
            model_claims = eval_json.get("model_total_claims", 1)
            supported = eval_json.get("model_supported_claims", 0)
            
            recall = recalled / max(1, gold_count)
            precision = supported / max(1, model_claims)
            premise = eval_json.get("premise_handling", "n/a")
            
            return precision, recall, eval_json.get("reasoning", ""), premise
        else:
            return 0.0, 0.0, "Failed to parse JSON", "n/a"
    except Exception as e:
        return 0.0, 0.0, f"Eval error: {str(e)}", "n/a"

def main():
    os.makedirs(SCORING_DIR, exist_ok=True)
    
    with open(GOLD_FILE, "r", encoding="utf-8") as f:
        gold_items = json.load(f)
        
    gold_map = {item["item_id"]: {"triplets": item.get("triplets", []), "premise": item.get("premise_handling", "")} for item in gold_items}
    
    output_files = glob.glob(os.path.join(OUTPUTS_DIR, "*.json"))
    
    results = []
    
    print(f"Scoring {len(output_files)} model responses using google/gemma-4-31B-it...")
    
    for file_path in output_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        item_id = data.get("item_id")
        model = data.get("model_version")
        response = data.get("response", "")
        
        gold_triplets = gold_map.get(item_id, {}).get("triplets", [])
        gold_premise = gold_map.get(item_id, {}).get("premise", "")
        
        print(f"Scoring {item_id} - {model}...")
        precision, recall, notes, premise = auto_score_with_llm(response, gold_triplets, gold_premise)
        
        results.append({
            "item_id": item_id,
            "model": model,
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "premise_handling": premise,
            "notes": notes
        })
        
    df = pd.DataFrame(results)
    
    csv_path = os.path.join(SCORING_DIR, "scores.csv")
    df.to_csv(csv_path, index=False)
    
    summary_path = os.path.join(SCORING_DIR, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Scoring Summary\n\n")
        
        agg = df.groupby("model")[["precision", "recall"]].mean().reset_index()
        f.write("## Aggregate Metrics\n")
        f.write(agg.to_markdown(index=False) + "\n\n")
        
        f.write("## Per-Item Results\n")
        f.write(df[["item_id", "model", "precision", "recall", "premise_handling"]].to_markdown(index=False) + "\n")
        
    print(f"Scoring complete! Saved to {csv_path} and {summary_path}")

if __name__ == "__main__":
    main()
