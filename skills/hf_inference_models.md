# HuggingFace Inference Models — Usage Reference

This document records the exact model IDs, API call patterns, and configuration
notes for all models used in this benchmark. Cited so future runs in this repo
don't need to re-research model docs.

---

## 1. google/gemma-4-31B-it (Scoring Model)

- **Model ID:** `google/gemma-4-31B-it`
- **Role:** Evaluator/Judge. Used in `score.py` to evaluate precision and recall of factual triplets.
- **Size:** Dense 31B parameters
- **License:** Check model card
- **Context:** 128K tokens
- **API Endpoint (HF Inference, OpenAI-compatible):**

```python
import requests

HF_TOKEN = "hf_..."
MODEL_ID = "google/gemma-4-31B-it"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}/v1/chat/completions"

response = requests.post(
    API_URL,
    headers={
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    },
    json={
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": "You are a clinical data quality auditor."},
            {"role": "user", "content": "Audit this record: ..."},
        ],
        "max_tokens": 1024,
        "temperature": 0.3,
    },
)
result = response.json()
text = result["choices"][0]["message"]["content"]
```

---

## 2. zai-org/GLM-5.2 (Z.ai) — Generation Model

- **Model ID:** `zai-org/GLM-5.2`
- **License:** MIT
- **Context:** Up to 1M tokens (long-horizon reasoning)
- **Serving options:**
  1. **HF Inference Endpoints** — deploy as an OpenAI-compatible endpoint
  2. **vLLM / SGLang** — self-host with OpenAI-compatible server

### HF Inference (OpenAI-compatible)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/hf-inference/models/zai-org/GLM-5.2/v1",
    api_key="hf_...",
)

response = client.chat.completions.create(
    model="zai-org/GLM-5.2",
    messages=[
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
    ],
    max_tokens=2048,
    temperature=0,  # deterministic for benchmark
)
text = response.choices[0].message.content
```

### vLLM Self-Hosting

```bash
python -m vllm.entrypoints.openai.api_server \
    --model zai-org/GLM-5.2 \
    --max-model-len 65536 \
    --tensor-parallel-size 2
```

Then use the same OpenAI client pointing to `http://localhost:8000/v1`.

---

## 3. Qwen/Qwen3.6-35B-A3B (Alibaba) — Scoring Model

- **Model ID:** `Qwen/Qwen3.6-35B-A3B`
- **License:** Apache 2.0
- **Context:** 262,144 tokens native
- **Architecture:** MoE, 35B total / ~3B active params

### Critical: Thinking Mode

By default, Qwen3.6 emits `<think>...</think>` reasoning traces before the final answer.

**For this benchmark:** We want direct grounded answers, NOT reasoning traces. Disable
thinking mode using one of these approaches:

#### Option A: Disable via API parameter (recommended)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/hf-inference/models/Qwen/Qwen3.6-35B-A3B/v1",
    api_key="hf_...",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3.6-35B-A3B",
    messages=[
        {"role": "user", "content": "..."},
    ],
    max_tokens=2048,
    temperature=0.7,
    top_p=0.8,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
        "top_k": 20,
        "presence_penalty": 1.5,
    },
)
text = response.choices[0].message.content
```

#### Option B: Parse out thinking traces (if thinking is left ON)

```python
import re

raw = response.choices[0].message.content
# Remove <think>...</think> block
final_answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
```

**For grounding scoring:** Parse only the post-think final answer unless explicitly
analyzing reasoning traces. The thinking block may contain speculative reasoning
that shouldn't be scored as factual claims.

### Recommended sampling (deterministic-ish grounding task)

```
temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5
```

These are non-thinking/instruct-leaning settings recommended by the model card for
tasks that want direct answers rather than extended reasoning.

---

## 4. deepseek-ai/DeepSeek-V4-Flash:deepinfra (Generation Model)

- **Model ID:** `deepseek-ai/DeepSeek-V4-Flash:deepinfra`
- **Role:** Generation model (evaluated)
- **Architecture:** MoE
- **Context:** Large Context Window

### HF Inference (OpenAI-compatible)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="hf_...",
)

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V4-Flash:deepinfra",
    messages=[
        {"role": "user", "content": "..."},
    ],
    max_tokens=2048,
    temperature=0,  # deterministic for benchmark
)
text = response.choices[0].message.content
```

### HF Inference (OpenAI-compatible)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/hf-inference/models/google/gemma-4-31B-it/v1",
    api_key="hf_...",
)

response = client.chat.completions.create(
    model="google/gemma-4-31B-it",
    messages=[
        {"role": "user", "content": "..."},
    ],
    max_tokens=2048,
    temperature=0,  # deterministic for benchmark
)
text = response.choices[0].message.content
```



---

## General Notes

- **Temperature 0** for all scoring model calls (deterministic benchmark).
  Exception: Qwen3.6 with thinking disabled uses temperature=0.7 per model card recommendation.
- **One call per item per model** — state this as a limitation in the report.
- **Record exact model version** from the API response headers or model card.
- **API vs Website:** State which interface was used per model in REPORT.md header.
- All calls go through HF Inference API using the OpenAI-compatible chat completions
  endpoint pattern: `https://router.huggingface.co/hf-inference/models/{MODEL_ID}/v1/chat/completions`
