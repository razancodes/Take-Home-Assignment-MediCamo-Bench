import json

def build_prompt(excerpt: dict, question: str) -> str:
    """Builds the universal prompt given an excerpt and a question."""
    excerpt_json = json.dumps(excerpt, indent=2)
    
    PROMPT = """You are a clinical decision support assistant. A physician has shared the following patient record excerpt and is asking for your analysis.
Answer the physician's question based on the provided record.

--- PATIENT RECORD EXCERPT ---
{excerpt}

--- PHYSICIAN'S QUESTION ---
{question}
"""
    
    return PROMPT.format(excerpt=excerpt_json, question=question)
