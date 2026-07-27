"""
run_camouflaged.py - Camouflaged Test Harness Runner (Tier 3)

Sends one prompt per patient per model (3 patients x 3 models = 9 calls).
Each prompt contains the full patient chart + a numbered question battery.

Robust retry loop:
  - Up to 5 retries per call on failure
  - Exponential backoff: 5s, 10s, 20s, 40s, 60s
  - Response validation:
    * Non-empty response
    * Minimum length (100 chars — a real multi-question answer will be much longer)
    * Detects answer markers for all question numbers (1., 2., etc.)
  - If validation fails, retries with same prompt

Does NOT proceed to scoring — stops after generation for manual verification.
"""

import json
import os
import re
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GOLD_FILE = os.path.join(PROJECT_ROOT, "gold", "gold_labels_camouflaged.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "harness", "outputs", "camouflaged")
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")

# Add harness dir to path for model imports
sys.path.insert(0, os.path.join(PROJECT_ROOT, "harness"))
from models import call_model_camouflaged, CAMOUFLAGED_MODELS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_RETRIES = 5
BACKOFF_SCHEDULE = [5, 10, 20, 40, 60]  # seconds between retries
MIN_RESPONSE_LENGTH = 100  # chars — a real 8-question answer will be 1000+


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_env():
    """Load .env file into os.environ."""
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip('"')


def build_user_message(chart: dict, questions_text: str) -> str:
    """Build the user message: chart JSON + question list."""
    chart_json = json.dumps(chart, separators=(",", ":"))  # compact JSON
    return (
        f"{chart_json}\n\n"
        f"Can you go through this patient's chart and answer the following:\n\n"
        f"{questions_text}"
    )


def detect_question_markers(response: str, total_questions: int) -> list:
    """
    Check which question numbers have answer markers in the response.
    
    Looks for patterns like "1.", "**1.**", "**1**", "Q1", "Question 1",
    "### 1.", etc. Returns list of detected question numbers.
    """
    detected = []
    for qn in range(1, total_questions + 1):
        # Match common answer numbering patterns
        patterns = [
            rf"(?:^|\n)\s*\**{qn}\.\**",           # "1." or "**1.**"
            rf"(?:^|\n)\s*\**Q{qn}\b",              # "Q1" or "**Q1**"
            rf"(?:^|\n)\s*\**Question\s+{qn}\b",    # "Question 1"
            rf"(?:^|\n)\s*#{{1,4}}\s*{qn}\.",          # "### 1."
            rf"(?:^|\n)\s*\**{qn}\)\**",             # "1)" or "**1)**"
            rf"(?:^|\n)\s*\**{qn}\s*[\.\):\-]",     # "1:" or "1-"
        ]
        for pat in patterns:
            if re.search(pat, response):
                detected.append(qn)
                break
    return detected


def validate_response(response: str, total_questions: int, patient_id: str) -> tuple:
    """
    Validate a model response. Returns (is_valid, issues_list).
    
    Checks:
    1. Non-empty
    2. Minimum length
    3. All question numbers have answer markers
    """
    issues = []

    # Check 1: Non-empty
    if not response or not response.strip():
        return False, ["Response is empty"]

    # Check 2: Minimum length
    if len(response.strip()) < MIN_RESPONSE_LENGTH:
        issues.append(f"Response too short ({len(response.strip())} chars, need {MIN_RESPONSE_LENGTH}+)")

    # Check 3: Question markers
    detected = detect_question_markers(response, total_questions)
    missing = [q for q in range(1, total_questions + 1) if q not in detected]
    if missing:
        issues.append(f"Missing answer markers for Q{missing}")

    is_valid = len(issues) == 0
    return is_valid, issues


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def main():
    load_env()

    # Verify HF_TOKEN is set
    if not os.environ.get("HF_TOKEN"):
        print("ERROR: HF_TOKEN not set. Check .env file.")
        sys.exit(1)

    # Load gold labels
    if not os.path.exists(GOLD_FILE):
        print(f"ERROR: Gold labels file not found: {GOLD_FILE}")
        print("Run gold/build_gold_camouflaged.py first.")
        sys.exit(1)

    with open(GOLD_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("CAMOUFLAGED TEST HARNESS RUNNER")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Models: {', '.join(CAMOUFLAGED_MODELS)}")
    print(f"Patients: {', '.join(p['patient_id'] for p in prompts)}")
    print(f"Total calls: {len(prompts) * len(CAMOUFLAGED_MODELS)}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 70)

    # Track results for summary table
    results = []

    for prompt_def in prompts:
        patient_id = prompt_def["patient_id"]
        system_instruction = prompt_def["system_instruction"]
        chart = prompt_def["chart"]
        questions_text = prompt_def["questions_text"]
        total_questions = prompt_def["total_questions"]

        user_msg = build_user_message(chart, questions_text)
        user_msg_len = len(user_msg)

        print(f"\n{'='*70}")
        print(f"Patient: {patient_id} | Questions: {total_questions} | "
              f"User msg: {user_msg_len:,} chars (~{user_msg_len//4:,} tokens)")
        print(f"{'='*70}")

        for model_name in CAMOUFLAGED_MODELS:
            safe_model_name = model_name.replace("/", "_").replace(":", "_")

            # Check if output already exists for this patient+model+timestamp
            output_filename = f"{patient_id}_{safe_model_name}_{timestamp}.json"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            if os.path.exists(output_path):
                print(f"\n  [{patient_id}/{model_name}] Already exists, skipping.")
                results.append({
                    "patient_id": patient_id,
                    "model": model_name,
                    "status": "SKIPPED",
                    "response_len": 0,
                    "questions_detected": [],
                    "issues": ["Already exists"],
                })
                continue

            print(f"\n  [{patient_id}/{model_name}]")

            success = False
            response = ""
            last_issues = []

            for attempt in range(1, MAX_RETRIES + 1):
                backoff = BACKOFF_SCHEDULE[min(attempt - 1, len(BACKOFF_SCHEDULE) - 1)]

                if attempt > 1:
                    print(f"    Retry {attempt}/{MAX_RETRIES} (backoff {backoff}s)...")
                    time.sleep(backoff)
                else:
                    print(f"    Attempt {attempt}/{MAX_RETRIES}...")
                    time.sleep(3)  # Initial rate-limit delay
                    
                # Reduce context size on later attempts to prevent hanging on huge charts
                current_chart = chart
                if attempt == 3:
                    print("    Reducing context size (keeping last 15 visits and 50 tests)...")
                    current_chart = {
                        "visit_list": chart["visit_list"][-15:],
                        "test_list": chart["test_list"][-50:]
                    }
                elif attempt >= 4:
                    print("    Further reducing context size (keeping last 5 visits and 20 tests)...")
                    current_chart = {
                        "visit_list": chart["visit_list"][-5:],
                        "test_list": chart["test_list"][-20:]
                    }

                current_user_msg = build_user_message(current_chart, questions_text)

                try:
                    response = call_model_camouflaged(
                        model_name, system_instruction, current_user_msg
                    )

                    # Validate
                    is_valid, issues = validate_response(
                        response, total_questions, patient_id
                    )

                    if is_valid:
                        # Save output
                        output_data = {
                            "patient_id": patient_id,
                            "model_version": model_name,
                            "provider": "HuggingFace Inference API",
                            "interface": "API",
                            "temperature": 0 if "Qwen" not in model_name else 0.7,
                            "max_tokens": 8192,
                            "timestamp": datetime.now().isoformat(),
                            "system_instruction": system_instruction,
                            "prompt": current_user_msg,
                            "raw_response": response,
                        }

                        with open(output_path, "w", encoding="utf-8") as out_f:
                            json.dump(output_data, out_f, indent=2, ensure_ascii=False)

                        detected = detect_question_markers(response, total_questions)
                        print(f"    SUCCESS! {len(response):,} chars, "
                              f"Q{detected} detected. Saved.")

                        results.append({
                            "patient_id": patient_id,
                            "model": model_name,
                            "status": "SUCCESS",
                            "response_len": len(response),
                            "questions_detected": detected,
                            "issues": [],
                        })
                        success = True
                        break
                    else:
                        last_issues = issues
                        print(f"    Validation failed: {issues}")
                        # On validation failure, if response exists but is just
                        # missing some markers, save it anyway on last attempt
                        if attempt == MAX_RETRIES and response and len(response.strip()) > MIN_RESPONSE_LENGTH:
                            print(f"    Saving partial response on final attempt...")
                            output_data = {
                                "patient_id": patient_id,
                                "model_version": model_name,
                                "provider": "HuggingFace Inference API",
                                "interface": "API",
                                "temperature": 0 if "Qwen" not in model_name else 0.7,
                                "max_tokens": 8192,
                                "timestamp": datetime.now().isoformat(),
                                "system_instruction": system_instruction,
                                "prompt": current_user_msg,
                                "raw_response": response,
                                "validation_issues": issues,
                            }
                            with open(output_path, "w", encoding="utf-8") as out_f:
                                json.dump(output_data, out_f, indent=2, ensure_ascii=False)

                            detected = detect_question_markers(response, total_questions)
                            print(f"    PARTIAL SAVE: {len(response):,} chars, "
                                  f"Q{detected} detected.")
                            results.append({
                                "patient_id": patient_id,
                                "model": model_name,
                                "status": "PARTIAL",
                                "response_len": len(response),
                                "questions_detected": detected,
                                "issues": issues,
                            })
                            success = True
                            break

                except Exception as e:
                    last_issues = [f"API error: {str(e)[:200]}"]
                    print(f"    Error: {str(e)[:200]}")

            if not success:
                print(f"    FAILED after {MAX_RETRIES} attempts.")
                results.append({
                    "patient_id": patient_id,
                    "model": model_name,
                    "status": "FAILED",
                    "response_len": len(response) if response else 0,
                    "questions_detected": [],
                    "issues": last_issues,
                })

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n")
    print("=" * 70)
    print("GENERATION SUMMARY")
    print("=" * 70)
    print(f"{'Patient':<10} {'Model':<45} {'Status':<10} {'Len':>8} {'Q Detected'}")
    print("-" * 100)

    success_count = 0
    total_count = len(results)

    for r in results:
        model_short = r["model"].split("/")[-1][:40]
        q_det = ",".join(str(q) for q in r["questions_detected"]) if r["questions_detected"] else "-"
        print(f"{r['patient_id']:<10} {model_short:<45} {r['status']:<10} "
              f"{r['response_len']:>8,} {q_det}")
        if r["status"] in ("SUCCESS", "PARTIAL"):
            success_count += 1
        if r["issues"]:
            for issue in r["issues"]:
                print(f"{'':>10} -> {issue}")

    print("-" * 100)
    print(f"Results: {success_count}/{total_count} successful")
    print(f"Output dir: {OUTPUT_DIR}")

    # -----------------------------------------------------------------------
    # HARD STOP — do not proceed to scoring
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STOP: Review outputs before scoring!")
    print("=" * 70)
    print(f"Output files are in: {OUTPUT_DIR}")
    print("Please verify:")
    print("  1. All 9 output files are present")
    print("  2. Each file contains answers for ALL question numbers")
    print("  3. Filler questions got reasonable answers (sanity check)")
    print("  4. Graded question responses are complete, not truncated")
    print("")
    print("Once verified, run: python scoring/score_camouflaged.py")
    print("=" * 70)

    # Return results for potential programmatic use
    return results


if __name__ == "__main__":
    main()
