"""Measures how many tokens a document group's content actually needs,
then picks a local Ollama model whose REAL trained context can hold it -
issue #7. Ported from the sibling `docuzent` (Rust) project's own
`docuzent_core::vram` module, which already found and fixed the exact
same traps this reuses the fix for, rather than re-deriving them:

- A model's *nominal* `context_length` can be a RoPE-scaling
  extrapolation far beyond what it was actually trained on (confirmed
  via `devstral-small-2:24b`: nominal 393,216, really trained on 8,192 -
  using the nominal number sized a call that crashed Ollama outright).
  This module always prefers the real trained context
  (`rope.scaling.original_context_length`) when a model reports one.
- KV-cache bytes per token is computed from real GGUF architecture
  metadata (block_count, attention.head_count_kv, attention.key_length),
  not guessed per model family - and `attention.key_length` is
  preferred over the derived `embedding_length / head_count`
  approximation, which was found to be measurably wrong (a real 25%
  overestimate for mistral3, a real 2.9x underestimate for gemma4).

Deliberately does NOT cap context to fit free VRAM, unlike the sibling
project's own default - real testing here found accuracy tracks whether
the actual content fits in context, so this prioritizes fitting the
real content over fitting VRAM: RAM offload is warned about, never used
as a reason to shrink the context.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import requests

# Same estimate the sibling docuzent (Rust) project uses, for the same
# reason: Ollama exposes no tokenizer endpoint to do better without
# bundling one. Good enough for a sizing decision, not exact accounting.
CHARS_PER_TOKEN = 4

# f16 is Ollama's default KV-cache element size (2 bytes/value) - see
# the sibling project's own vram.rs for why this is a real, documented
# assumption (a server configured for quantized KV cache would have a
# smaller real footprint than this estimates - a conservative bias).
KV_CACHE_DTYPE_BYTES = 2

# Response + prompt-template overhead reserved on top of the raw content
# token estimate - never plan to fill a context window to its exact
# limit.
OVERHEAD_TOKENS = 2000


@dataclass
class ModelContextInfo:
    name: str
    nominal_context_length: int
    original_context_length: int | None  # real trained context, when reported
    weight_bytes: int
    kv_bytes_per_token: int

    @property
    def trained_context_length(self) -> int:
        """The number to actually trust - the real trained context when
        known, the nominal window otherwise. Never the other way around."""
        return self.original_context_length or self.nominal_context_length


def estimate_required_tokens(text: str) -> int:
    """How many tokens `text` needs, plus a fixed overhead margin for
    the prompt template and response - an estimate, not exact token
    accounting (see module docstring)."""
    return (len(text) // CHARS_PER_TOKEN) + OVERHEAD_TOKENS


def _avg_of(value) -> float | None:
    """A GGUF metadata field can be one scalar or a per-layer array (some
    architectures vary it per layer, e.g. Gemma's interleaved local/
    global attention) - averaging an array and multiplying by layer
    count elsewhere reproduces the true total exactly."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and value:
        nums = [v for v in value if isinstance(v, (int, float))]
        if nums:
            return sum(nums) / len(nums)
    return None


def _find_by_suffix(model_info: dict, suffix: str):
    for key, value in model_info.items():
        if key.endswith(suffix):
            return value
    return None


def _model_context_info(host: str, name: str, weight_bytes: int) -> ModelContextInfo | None:
    resp = requests.post(f"{host.rstrip('/')}/api/show", json={"model": name}, timeout=30)
    if not resp.ok:
        return None
    model_info = resp.json().get("model_info", {})

    nominal = _find_by_suffix(model_info, ".context_length")
    if not nominal:
        return None
    original = _find_by_suffix(model_info, ".rope.scaling.original_context_length")

    num_layers = _avg_of(_find_by_suffix(model_info, ".block_count"))
    num_kv_heads = _avg_of(_find_by_suffix(model_info, ".attention.head_count_kv"))
    head_dim = _avg_of(_find_by_suffix(model_info, ".attention.key_length"))
    if head_dim is None:
        # Fallback matching the sibling project's own documented
        # approximation (embedding_length / head_count) - used only when
        # a model doesn't report its real per-head dimension directly.
        embedding_length = _avg_of(_find_by_suffix(model_info, ".embedding_length"))
        head_count = _avg_of(_find_by_suffix(model_info, ".attention.head_count"))
        head_dim = (embedding_length / head_count) if embedding_length and head_count else None

    kv_bytes_per_token = 0
    if num_layers and num_kv_heads and head_dim:
        kv_bytes_per_token = round(2 * num_layers * num_kv_heads * head_dim * KV_CACHE_DTYPE_BYTES)

    return ModelContextInfo(
        name=name,
        nominal_context_length=int(nominal),
        original_context_length=int(original) if original else None,
        weight_bytes=weight_bytes,
        kv_bytes_per_token=kv_bytes_per_token,
    )


def list_model_context_info(host: str) -> list[ModelContextInfo]:
    """Real context/KV-cache info for every model this Ollama server has
    locally pulled - one `/api/show` call per model (no tokenizer
    endpoint exists to do this more cheaply)."""
    resp = requests.get(f"{host.rstrip('/')}/api/tags", timeout=30)
    resp.raise_for_status()
    models = resp.json().get("models", [])

    infos = []
    for m in models:
        info = _model_context_info(host, m["name"], m.get("size", 0))
        if info:
            infos.append(info)
    return infos


def vram_free_bytes() -> int | None:
    """Real free VRAM via `nvidia-smi` - NVIDIA-only, `None` on any other
    GPU vendor or if `nvidia-smi` isn't on PATH, same limitation the
    sibling project documents for the same underlying command."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        used_str, total_str = result.stdout.strip().splitlines()[0].split(",")
        used, total = int(used_str.strip()), int(total_str.strip())
        return (total - used) * 1024 * 1024  # nvidia-smi reports MiB
    except (ValueError, IndexError):
        return None


@dataclass
class ModelSelection:
    model: ModelContextInfo
    required_tokens: int
    fits: bool
    warning: str | None
    ram_offload_warning: str | None


def select_model(candidates: list[ModelContextInfo], required_tokens: int, free_vram_bytes: int | None) -> ModelSelection:
    """The pure decision logic, independent of any network/subprocess
    call - given a real (or test-supplied) candidate list, required
    token count, and free-VRAM reading, decides which model to use.
    Separated from `pick_model_for_tokens` so this can be unit-tested
    directly against fabricated data, matching this project's existing
    "no live model needed" test convention.

    Picks the smallest available model whose real trained context
    comfortably holds `required_tokens`. If none do, warns clearly and
    falls back to whichever model has the largest real trained context -
    best effort, never a hard failure. Also checks (but never acts on)
    whether the chosen model+context would need to spill into system RAM."""
    if not candidates:
        raise RuntimeError("no models with readable context info were found - is `ollama serve` running with at least one model pulled?")

    fitting = sorted((c for c in candidates if c.trained_context_length >= required_tokens), key=lambda c: c.trained_context_length)

    if fitting:
        chosen = fitting[0]
        warning = None
        fits = True
    else:
        chosen = max(candidates, key=lambda c: c.trained_context_length)
        warning = (
            f"No locally available model has a real trained context >= {required_tokens} tokens "
            f"(the largest is `{chosen.name}` at {chosen.trained_context_length}). "
            f"Proceeding with `{chosen.name}` as the best available option - the document group's content "
            f"will not fully fit, which real testing shows correlates with worse results."
        )
        fits = False

    ram_offload_warning = None
    if free_vram_bytes is not None and chosen.kv_bytes_per_token > 0:
        estimated_bytes = chosen.weight_bytes + chosen.kv_bytes_per_token * min(required_tokens, chosen.trained_context_length)
        if estimated_bytes > free_vram_bytes:
            ram_offload_warning = (
                f"`{chosen.name}` at this context size is estimated to need ~{estimated_bytes / 1e9:.1f} GB, "
                f"more than the ~{free_vram_bytes / 1e9:.1f} GB of free VRAM detected - it will likely spill into system RAM "
                f"(slower, not blocked). Proceeding anyway: fitting the real content in context matters more here "
                f"than avoiding RAM offload."
            )

    return ModelSelection(model=chosen, required_tokens=required_tokens, fits=fits, warning=warning, ram_offload_warning=ram_offload_warning)


def pick_model_for_tokens(required_tokens: int, host: str) -> ModelSelection:
    """Real, live version: queries Ollama for real candidates and real
    free VRAM, then applies `select_model`'s decision logic."""
    candidates = list_model_context_info(host)
    return select_model(candidates, required_tokens, vram_free_bytes())
