
import pytest
import subprocess
import sys

def run_cli(args):
    """Run the sdeval CLI with given arguments."""
    cmd = ["sdeval"] + args
    # We use subprocess to test the actual installed CLI, 
    # but since we might not have re-installed yet, we can also invoke via python -m sdeval.cli
    # For TDD, let's assume we invoke via python module to test the code directly.
    cmd = [sys.executable, "-m", "sdeval.cli"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

def test_dp_help():
    """Test that 'dp' command exists and shows help."""
    result = run_cli(["dp", "--help"])
    # This should fail initially as 'dp' command is not added yet
    if result.returncode != 0:
        pytest.fail(f"CLI failed: {result.stderr}")
    assert "Differential Privacy" in result.stdout or "dp" in result.stdout

def test_dp_gaussian():
    """Test gaussian mechanism CLI."""
    result = run_cli(["dp", "gaussian", "--value", "100", "--sigma", "1.0"])
    if result.returncode != 0:
        pytest.fail(f"CLI failed: {result.stderr}")
    assert "Noisy count" in result.stdout or "result" in result.stdout

def test_dp_conversion():
    """Test privacy conversion CLI."""
    result = run_cli(["dp", "cdp-to-delta", "--rho", "0.5", "--epsilon", "1.0"])
    if result.returncode != 0:
        pytest.fail(f"CLI failed: {result.stderr}")
    assert "delta" in result.stdout
