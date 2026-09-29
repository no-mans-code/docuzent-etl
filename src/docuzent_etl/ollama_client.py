"""A minimal, synchronous client for Ollama's `/api/generate` - deliberately
small, mirroring the sibling `docuzent` (Rust) project's own client rather
than pulling in a heavier framework for one HTTP call shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import requests

DEFAULT_NUM_PREDICT = 4096  # generous: schema proposals and generated code can be long


@dataclass
class GenerateResult:
    response: str
    model: str
    prompt_eval_count: int
    eval_count: int


def generate(host: str, model: str, prompt: str, num_predict: int = DEFAULT_NUM_PREDICT, timeout_s: float = 300.0) -> GenerateResult:
    """One blocking call to `/api/generate`, non-streaming - this
    prototype's calls are all "wait for the whole schema/script," not
    something a human is watching token-by-token."""
    resp = requests.post(
        f"{host.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": num_predict},
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    return GenerateResult(
        response=data.get("response", ""),
        model=model,
        prompt_eval_count=data.get("prompt_eval_count", 0),
        eval_count=data.get("eval_count", 0),
    )


def extract_json_block(text: str) -> str:
    """Models routinely wrap JSON in markdown code fences or add prose
    before/after it, even when explicitly asked for "JSON only" - this
    finds the first `{...}` or `[...]` block by bracket matching rather
    than assuming the whole response is clean JSON, which real testing
    showed is not a safe assumption."""
    text = text.strip()
    # Strip a markdown fence if the whole response is wrapped in one.
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start_chars = "{["
    end_chars = {"{": "}", "[": "]"}
    for i, ch in enumerate(text):
        if ch in start_chars:
            depth = 0
            closing = end_chars[ch]
            for j in range(i, len(text)):
                if text[j] == ch:
                    depth += 1
                elif text[j] == closing:
                    depth -= 1
                    if depth == 0:
                        return text[i : j + 1]
            break  # unbalanced - fall through to returning the raw text
    return text


def parse_json_response(text: str) -> tuple[object | None, str | None]:
    """Returns `(parsed, None)` on success or `(None, error_message)` on
    failure - never raises, since a malformed model response is an
    expected, retry-able outcome here, not a bug."""
    candidate = extract_json_block(text)
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
