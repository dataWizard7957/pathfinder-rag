import json
import requests
from typing import List, Dict, Any

from src.config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL
)


def _extract_json_object(text: str) -> str:
    """Extract the first top-level JSON object from a string."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return text[start : end + 1]


def _llm_generate(prompt: str, timeout: int = 300, force_json: bool = False) -> str:
    """
    Low-level LLM call.
    Provider chosen via env/config: LLM_PROVIDER=ollama|groq
    Returns plain text content produced by the model.
    """
    provider = (LLM_PROVIDER or "ollama").lower()

    if provider == "ollama":
        payload: Dict[str, Any] = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "temperature": 0,
            # speed + stability options 
            "keep_alive": "10m",
            "options": {
                "num_predict": 450,  # cap output tokens 
                "num_ctx": 2048,     # smaller context window (faster)
            },
        }
        if force_json:
            payload["format"] = "json"

        r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        return (r.json().get("response") or "").strip()

    raise RuntimeError(f"Unsupported LLM_PROVIDER: {provider}")


def _repair_json_with_llm(bad_text: str) -> str:
    """Formatting-only JSON repair using the LLM."""
    prompt = f"""
Fix the following into VALID JSON.

Rules:
- Output ONLY valid JSON (no markdown, no commentary).
- Keep exactly these keys only: suggestions, follow_up_questions
- suggestions must be a JSON array of exactly 5 objects, each with keys: title, why
- follow_up_questions must be a JSON array of strings (3 to 7 items)
- Use double quotes for all strings.
- Escape any internal quotes inside strings.

INPUT (may be invalid JSON):
{bad_text}
""".strip()

    # One repair attempt (keep it fast)
    return _llm_generate(prompt, timeout=120, force_json=False)


def generate_with_llm(context_chunks: List[Dict], age_range: str, income_type: str) -> Dict[str, Any]:
    """
    LLM-backed grounded generation.
    Returns:
    {
      "suggestions": [{"title": "...", "why": "..."}, ... x5],
      "follow_up_questions": ["...", ...]
    }
    """

    # Keep prompts stable and prevent timeouts
    context_chunks = context_chunks[:3]   # smaller context => faster
    MAX_CHUNK_CHARS = 600                

    context_text = "\n\n".join(
        f"[{c.get('chunk_id', '')}]\n{(c.get('text') or '')[:MAX_CHUNK_CHARS]}"
        for c in context_chunks
    )

    schema_hint = {
        "suggestions": [{"title": "string", "why": "string"}],
        "follow_up_questions": ["string"],
    }

    prompt = f"""
You are a mortgage broker assistant.
Use ONLY the information in the CONTEXT. Do not add external rules.

Return ONLY valid JSON. No markdown. No headings. No extra keys.
Use double quotes for all JSON strings. Escape internal quotes.

Profile:
age_range: {age_range}
income_type: {income_type}

CONTEXT:
{context_text}

JSON schema example (types only):
{json.dumps(schema_hint)}

Requirements:
- suggestions: exactly 5 items
- follow_up_questions: 3 to 7 questions
- Ground everything in the context
""".strip()

    raw = _llm_generate(prompt, timeout=300, force_json=True)

    # Parse attempt 1
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Parse attempt 2 (extract object)
    try:
        extracted = _extract_json_object(raw)
        return json.loads(extracted)
    except Exception:
        pass

    # Repair attempt
    repaired = _repair_json_with_llm(raw)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        extracted = _extract_json_object(repaired)
        return json.loads(extracted)
