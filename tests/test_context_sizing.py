import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docuzent_etl.context_sizing import (  # noqa: E402
    ModelContextInfo,
    _avg_of,
    _find_by_suffix,
    estimate_required_tokens,
    select_model,
)


def test_estimate_required_tokens_scales_with_length_and_adds_overhead():
    short = estimate_required_tokens("a" * 400)
    long = estimate_required_tokens("a" * 4000)
    assert long > short
    assert short >= 2000  # the fixed overhead margin alone


def test_avg_of_handles_a_plain_scalar():
    assert _avg_of(40) == 40.0


def test_avg_of_averages_a_per_layer_array():
    """Real case this exists for: gemma4's interleaved local/global
    attention reports head_count_kv as a per-layer array, not one
    scalar - confirmed via the sibling docuzent (Rust) project's own
    real /api/show payload."""
    assert _avg_of([8, 8, 8, 2]) == 6.5


def test_avg_of_returns_none_for_unusable_input():
    assert _avg_of(None) is None
    assert _avg_of([]) is None
    assert _avg_of("not a number") is None


def test_find_by_suffix_ignores_the_family_prefix():
    model_info = {"mistral3.context_length": 393216, "mistral3.embedding_length": 5120}
    assert _find_by_suffix(model_info, ".context_length") == 393216


def test_find_by_suffix_returns_none_when_nothing_matches():
    assert _find_by_suffix({"a.b": 1}, ".context_length") is None


def test_trained_context_length_prefers_the_real_trained_value():
    """The real lesson this exists to encode: devstral-small-2:24b
    reports a nominal 393,216-token context but was actually trained on
    8,192 - trusting the nominal number crashed Ollama outright in the
    sibling project's own real testing. Never use nominal when a real
    trained value is known."""
    info = ModelContextInfo(name="devstral", nominal_context_length=393216, original_context_length=8192, weight_bytes=0, kv_bytes_per_token=0)
    assert info.trained_context_length == 8192


def test_trained_context_length_falls_back_to_nominal_when_no_scaling_reported():
    info = ModelContextInfo(name="qwen", nominal_context_length=32768, original_context_length=None, weight_bytes=0, kv_bytes_per_token=0)
    assert info.trained_context_length == 32768


def _model(name, trained_context, weight_bytes=1_000_000_000, kv_bytes_per_token=100_000):
    return ModelContextInfo(name=name, nominal_context_length=trained_context, original_context_length=None, weight_bytes=weight_bytes, kv_bytes_per_token=kv_bytes_per_token)


def test_select_model_picks_the_smallest_model_that_actually_fits():
    """Real rationale: once a model's context comfortably holds the
    content, a bigger context buys nothing further for this decision -
    prefer the cheaper model, matching the issue's own "does well if the
    data fits" framing (fitting is what matters, not maximizing size)."""
    candidates = [_model("small", 8192), _model("medium", 32768), _model("large", 131072)]
    selection = select_model(candidates, required_tokens=10000, free_vram_bytes=None)
    assert selection.model.name == "medium"
    assert selection.fits is True
    assert selection.warning is None


def test_select_model_falls_back_to_the_largest_when_nothing_fits():
    candidates = [_model("small", 4096), _model("medium", 8192)]
    selection = select_model(candidates, required_tokens=50000, free_vram_bytes=None)
    assert selection.model.name == "medium"
    assert selection.fits is False
    assert selection.warning is not None
    assert "medium" in selection.warning


def test_select_model_raises_a_clear_error_with_no_candidates_at_all():
    try:
        select_model([], required_tokens=1000, free_vram_bytes=None)
        assert False, "expected a RuntimeError"
    except RuntimeError as e:
        assert "ollama serve" in str(e)


def test_select_model_warns_about_ram_offload_without_blocking():
    """The core, deliberate difference from the sibling docuzent (Rust)
    project's own default: this must still pick the model and proceed,
    never shrink the context or refuse, when RAM offload is predicted."""
    tiny_free_vram = 500_000_000  # far less than the model would need
    candidates = [_model("big-model", 32768, weight_bytes=5_000_000_000, kv_bytes_per_token=200_000)]
    selection = select_model(candidates, required_tokens=10000, free_vram_bytes=tiny_free_vram)
    assert selection.model.name == "big-model"  # still chosen and proceeded with
    assert selection.ram_offload_warning is not None


def test_select_model_no_ram_warning_when_it_comfortably_fits_vram():
    huge_free_vram = 1_000_000_000_000  # 1 TB - obviously enough
    candidates = [_model("small-model", 8192, weight_bytes=1_000_000_000, kv_bytes_per_token=1000)]
    selection = select_model(candidates, required_tokens=4000, free_vram_bytes=huge_free_vram)
    assert selection.ram_offload_warning is None


def test_select_model_no_ram_warning_when_vram_cannot_be_measured():
    """No NVIDIA GPU detected (or nvidia-smi unavailable) must not be
    treated as "definitely will offload" - it's an unknown, not a
    warning-worthy fact."""
    candidates = [_model("some-model", 8192)]
    selection = select_model(candidates, required_tokens=4000, free_vram_bytes=None)
    assert selection.ram_offload_warning is None
