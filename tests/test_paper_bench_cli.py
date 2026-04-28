"""Step-0 plumbing tests: every paper bench exposes the new CLI flags.

Argparse-only smoke test — no model load, no CUDA, no datasets.
We import each bench's argparse-building code path and verify the
required flags are declared with the expected defaults / required-ness.

This catches the regression where v_tolerance was unsettable from the CLI.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

PAPER_BENCHES = ["pg19_perplexity", "niah", "ruler"]

# Flags every paper bench MUST expose after Step 0.
REQUIRED_FLAGS = [
    "--v-tolerance",
    "--use-int4-values",
    "--group-size",
    "--tau-cov",
    "--k-min",
    "--k-max",
    "--ranking-fallback",
    "--ranking-r",
    "--ranking-fallback-mode",
    "--score-consistency-check",
    "--eps-guard",
    "--exploration-rate",
    "--rung1-threshold",
    "--rung1-multiplier",
]


@pytest.mark.parametrize("bench", PAPER_BENCHES)
def test_paper_bench_help_exposes_required_flags(bench: str):
    """`<bench>.py --help` lists every required §7 flag."""
    script = REPO / "benchmarks" / "paper" / f"{bench}.py"
    result = subprocess.run(
        [PYTHON, str(script), "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"{bench} --help exited {result.returncode}:\n{result.stderr}"
    )
    help_text = result.stdout
    missing = [f for f in REQUIRED_FLAGS if f not in help_text]
    assert not missing, (
        f"{bench} is missing required flags: {missing}\n"
        f"Help text:\n{help_text}"
    )


@pytest.mark.parametrize("bench", PAPER_BENCHES)
def test_paper_bench_rejects_missing_v_tolerance(bench: str):
    """Running a paper bench without --v-tolerance fails at argparse.

    No silent default: reviewer-facing runs must record the paper value.
    """
    script = REPO / "benchmarks" / "paper" / f"{bench}.py"
    # Pass --output to a /tmp path so we don't write into the repo if the
    # bench somehow proceeds past argparse (it shouldn't).
    result = subprocess.run(
        [PYTHON, str(script), "--output", "/tmp/_test_should_not_exist.json"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0, (
        f"{bench} accepted invocation without --v-tolerance — should have rejected"
    )
    # argparse error message contains "v-tolerance" (or "v_tolerance")
    err = (result.stderr + result.stdout).lower()
    assert "v-tolerance" in err or "v_tolerance" in err, (
        f"{bench} rejected the invocation but error didn't mention v-tolerance:\n"
        f"{result.stderr}"
    )
