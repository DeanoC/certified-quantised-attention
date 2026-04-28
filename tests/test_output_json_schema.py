"""Step-0 plumbing tests: cache_config provenance helper.

Verifies the cache-config block that every paper bench embeds in its
output JSON contains the load-bearing fields needed to prove which
quantisation config produced the run, and that the hash is reproducible.

CPU-only — pure helper invocation, no kernels.
"""

import argparse
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmarks" / "paper"))

from _provenance import (  # noqa: E402
    add_paper_cache_args,
    add_paper_section7_args,
    cache_config_dict,
)


# Fields every paper bench's cache_config block MUST contain.
REQUIRED_FIELDS = {
    "v_tolerance",
    "quantization_mode",
    "asymmetric_keys",
    "use_int4_values",
    "group_size",
    "score_consistency_check",
    "tau_cov",
    "k_min",
    "k_max",
    "ranking_fallback",
    "ranking_r",
    "ranking_fallback_mode",
    "eps_guard",
    "exploration_rate",
    "rung1_threshold",
    "rung1_multiplier",
    "code_sha",
    "dotcache_config_hash",
}


def _build_args(**overrides):
    """Build a parsed-args namespace using the same flag wiring as a real bench."""
    p = argparse.ArgumentParser()
    add_paper_section7_args(p)
    add_paper_cache_args(p)
    base = ["--v-tolerance", "0.05"]
    for k, v in overrides.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                base.append(flag)
        else:
            base.extend([flag, str(v)])
    return p.parse_args(base)


def test_cache_config_dict_contains_all_required_fields():
    """Provenance block carries every paper-relevant setting."""
    args = _build_args(
        use_int4_values=True, group_size=16, tau_cov=0.995,
        score_consistency_check=True, ranking_fallback=True,
    )
    config = cache_config_dict(args)
    missing = REQUIRED_FIELDS - set(config.keys())
    assert not missing, f"Missing required fields: {sorted(missing)}"


def test_cache_config_paper_quantisation_mode():
    """With --use-int4-values --group-size 16, the mode string is canonical."""
    args = _build_args(use_int4_values=True, group_size=16)
    config = cache_config_dict(args)
    assert config["quantization_mode"] == "asymmetric_int8_keys+int4_g16_values"
    assert config["asymmetric_keys"] is True
    assert config["use_int4_values"] is True
    assert config["group_size"] == 16


def test_cache_config_legacy_fp16_values_mode():
    """Without --use-int4-values, mode reflects FP16 values."""
    args = _build_args()  # no --use-int4-values
    config = cache_config_dict(args)
    assert config["quantization_mode"] == "asymmetric_int8_keys+fp16_values"
    assert config["use_int4_values"] is False
    assert config["group_size"] is None


def test_cache_config_hash_reproducible_for_same_args():
    """Same flags → same dotcache_config_hash."""
    args1 = _build_args(use_int4_values=True, tau_cov=0.995)
    args2 = _build_args(use_int4_values=True, tau_cov=0.995)
    h1 = cache_config_dict(args1)["dotcache_config_hash"]
    h2 = cache_config_dict(args2)["dotcache_config_hash"]
    assert h1 == h2


def test_cache_config_hash_changes_when_v_tolerance_changes():
    """Different v_tolerance → different hash. Catches silent config drift."""
    p = argparse.ArgumentParser()
    add_paper_section7_args(p)
    add_paper_cache_args(p)
    a05 = p.parse_args(["--v-tolerance", "0.05"])
    p2 = argparse.ArgumentParser()
    add_paper_section7_args(p2)
    add_paper_cache_args(p2)
    a50 = p2.parse_args(["--v-tolerance", "0.5"])
    h_paper = cache_config_dict(a05)["dotcache_config_hash"]
    h_legacy = cache_config_dict(a50)["dotcache_config_hash"]
    assert h_paper != h_legacy, (
        "v_tolerance change must alter the config hash, otherwise the "
        "reviewer-facing provenance would miss config drift."
    )


def test_cache_config_v_tolerance_is_float():
    args = _build_args()
    config = cache_config_dict(args)
    assert isinstance(config["v_tolerance"], float)
    assert config["v_tolerance"] == 0.05


def test_section7_defaults_match_paper_operating_point():
    args = _build_args(use_int4_values=True)
    config = cache_config_dict(args)
    assert config["tau_cov"] == 0.995
    assert config["k_min"] == 2
    assert config["k_max"] == 128
    assert config["ranking_fallback"] is True
    assert config["ranking_r"] == 1
    assert config["ranking_fallback_mode"] == "full"
    assert config["eps_guard"] == 0.01
    assert config["exploration_rate"] == 0.02
    assert config["rung1_threshold"] == 0.02
    assert config["rung1_multiplier"] == 2.0
