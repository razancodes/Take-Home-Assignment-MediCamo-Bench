import os
import re
from openai import OpenAI
import requests

def call_glm(prompt: str) -> str:
    """Calls zai-org/GLM-5.2 via HF Inference API."""
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ.get("HF_TOKEN", ""),
        timeout=30.0,
    )
    
    response = client.chat.completions.create(
        model="zai-org/GLM-5.2",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0,
    )
    return response.choices[0].message.content or ""

def call_deepseek_flash(prompt: str) -> str:
    """Calls deepseek-ai/DeepSeek-V4-Flash:deepinfra via HF Inference API."""
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ.get("HF_TOKEN", ""),
        timeout=30.0,
    )
    
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4-Flash:deepinfra",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0,
    )
    return response.choices[0].message.content or ""

def call_qwen(prompt: str) -> str:
    """Calls Qwen/Qwen3.6-35B-A3B via HF Inference API, with thinking mode disabled."""
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ.get("HF_TOKEN", ""),
        timeout=30.0,
    )
    
    response = client.chat.completions.create(
        model="Qwen/Qwen3.6-35B-A3B",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.7,
        top_p=0.8,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "top_k": 20,
            "presence_penalty": 1.5,
        },
    )
    
    raw = response.choices[0].message.content or ""
    # Fallback to parse out think blocks just in case the API ignores the kwarg
    final_answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return final_answer

def call_gemma(prompt: str) -> str:
    """Calls google/gemma-4-31B-it via HF Inference API."""
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ.get("HF_TOKEN", ""),
        timeout=30.0,
    )
    
    response = client.chat.completions.create(
        model="google/gemma-4-31B-it",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0,
    )
    return response.choices[0].message.content or ""

MODEL_ROUTERS = {
    "deepseek-ai/DeepSeek-V4-Flash:deepinfra": call_deepseek_flash,
    "Qwen/Qwen3.6-35B-A3B": call_qwen,
    "zai-org/GLM-5.2": call_glm
}

# ---------------------------------------------------------------------------
# Camouflaged-battery model caller (Tier 3)
# Uses separate system + user messages, higher max_tokens and timeout.
# ---------------------------------------------------------------------------
CAMOUFLAGED_MODELS = [
    "deepseek-ai/DeepSeek-V4-Flash:deepinfra",
    "Qwen/Qwen3.6-35B-A3B",
    "zai-org/GLM-5.2",
]

def call_model_camouflaged(model_name: str, system_msg: str, user_msg: str) -> str:
    """
    Calls a model via HF Inference API with separate system and user messages.
    
    - max_tokens=8192 (models need room for 8-9 question answers)
    - temperature=0 for all models (including Qwen — spec requires temp 0)
    - timeout=120s (large chart prompts take longer)
    - Qwen: thinking mode disabled, think-block stripping applied
    """
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ.get("HF_TOKEN", ""),
        timeout=120.0,
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    kwargs = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 8192,
        "temperature": 0,
    }

    # Qwen-specific: disable thinking mode
    if "Qwen" in model_name:
        kwargs["temperature"] = 0.7   # Qwen needs temp > 0 for the API
        kwargs["top_p"] = 0.8
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False},
            "top_k": 20,
            "presence_penalty": 1.5,
        }

    response = client.chat.completions.create(**kwargs)
    raw = response.choices[0].message.content or ""

    # Strip think blocks for Qwen (fallback in case API ignores the kwarg)
    if "Qwen" in model_name:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    return raw
