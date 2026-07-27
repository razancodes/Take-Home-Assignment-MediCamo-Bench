import os
import pandas as pd
import glob
import json

SCORING_DIR = r"c:\Users\MRaza\Documents\TH-Assignment-AI-data\clinical-llm-benchmark\scoring"
OUTPUTS_DIR = r"c:\Users\MRaza\Documents\TH-Assignment-AI-data\clinical-llm-benchmark\harness\outputs"
REPORT_PATH = r"c:\Users\MRaza\Documents\TH-Assignment-AI-data\clinical-llm-benchmark\REPORT.md"

TRANSCRIPT_ITEMS = ["V-P0010-V00147", "L-P0006"]

def generate_report():
    scores_path = os.path.join(SCORING_DIR, "scores.csv")
    
    if os.path.exists(scores_path):
        df = pd.read_csv(scores_path)
        agg_md = df.groupby("model")[["precision", "recall"]].mean().reset_index().to_markdown(index=False)
        per_item_md = df[["item_id", "model", "precision", "recall", "premise_handling"]].to_markdown(index=False)
    else:
        agg_md = "*(Scores not generated yet)*"
        per_item_md = "*(Scores not generated yet)*"
        
    report = f"""# Clinical LLM Micro-Benchmark Report

## Model Details
* **deepseek-ai/DeepSeek-V4-Flash:deepinfra** - HF Inference API
* **Qwen/Qwen3.6-35B-A3B** - HF Inference API
* **zai-org/GLM-5.2** - HF Inference API

## 1. Data Audit Findings
*(Prose to be filled by user)*

## 2. Task Selection: Grounding & Hallucination
*(Prose to be filled by user)*

## 3. Method
*(Prose to be filled by user)*

## 4. Results

## 4. Results (Realistic Tier)

### Aggregate Performance
{agg_md}

### Per-Item Performance
{per_item_md}

## 5. Threats to Validity
*(Prose to be filled by user)*

## 6. AI Usage Note & Methods Citations
- **False premise questions:** KG-FPQ (Zhu et al., 2024); DecoPrompt (2024); "Identifying and Answering Questions with False Assumptions" (Wang & Blanco, 2025).
- **Long context / lost in the middle:** Liu et al., "Lost in the Middle" (2023).
- **Abstention:** AbstentionBench (NeurIPS 2025); "Two Axes of LLM Abstention" (2026).
- **Structured-output forced completion:** PhantomFill (2026).

---

## Appendix: Verbatim Transcripts
"""
    
    # Read outputs for the chosen items
    outputs = glob.glob(os.path.join(OUTPUTS_DIR, "*.json"))
    
    for item_id in TRANSCRIPT_ITEMS:
        report += f"\n### Transcripts for Item: {item_id}\n\n"
        for out_file in sorted(outputs):
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if data["item_id"] == item_id:
                model = data["model_version"]
                provider = data["provider"]
                interface = data["interface"]
                prompt = data["prompt"]
                response = data["response"]
                
                report += f"#### Model: {model}\n"
                report += f"**Provider/Interface**: {provider} / {interface}\n\n"
                report += "**Prompt:**\n```\n" + prompt + "\n```\n\n"
                report += "**Response:**\n```\n" + response + "\n```\n\n"
                report += "---\n\n"
                
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"Generated {REPORT_PATH}")

if __name__ == "__main__":
    generate_report()
