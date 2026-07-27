# MediCamo-Bench: Clinical LLM Benchmark

## Overview
**MediCamo-Bench** is a specialized benchmark designed to evaluate Large Language Models (LLMs) on their ability to handle messy, real-world Electronic Health Records (EHR). Unlike traditional benchmarks that test models on clean, idealized medical data, MediCamo-Bench tests for clinical safety, hallucination resistance, and the ability to detect contradictions in raw, unstructured patient data.

The core of this benchmark is the **Camouflaged Test**, which intentionally embeds "traps" (e.g., false premises, contradictory diagnoses, and missing critical information) to see if models will confidently hallucinate an answer or safely refuse to answer.

## Repository Structure

### Documentation & Reports
* [`REPORT.md`](REPORT.md) - The primary project write-up detailing the methodology, LLM performance results, and insights.
* [`TRANSCRIPTS.md`](TRANSCRIPTS.md) - Verbatim raw outputs from the evaluated models demonstrating how they interact with the embedded traps.
* [`gold/GOLD_LABELS.md`](gold/GOLD_LABELS.md) - Detailed documentation of the "camouflaged traps" designed for the benchmark.

### Core Pipelines
* **`audit/`** - Contains the data auditing pipeline used to sanitize and verify the raw EHR data before it is fed to the models. Highlights include the `AUDIT_SUMMARY.md` which documents the methodology used to handle unstructured fields.
* **`harness/`** - The evaluation harness responsible for formatting the clinical prompts and interacting with the open-weights models (Qwen, DeepSeek, GLM) via DeepInfra APIs.
* **`scoring/`** - Scripts for evaluating the model outputs. This includes `score_llm_judge.py`, which uses an LLM-as-a-judge to evaluate answers against the gold standard labels.

### Scripts
* [`generate_report.py`](generate_report.py) - A utility script to automate the generation of the final `REPORT.md` based on model outputs and evaluation scores.
* [`requirements.txt`](requirements.txt) - Python dependencies required to run the harness and scoring pipelines.
