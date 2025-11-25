
import pytest
import torch
import numpy as np

# These imports should fail initially
try:
    from sdeval.dp import (
        gaussian_mechanism,
        laplace_mechanism,
        exponential_mechanism,
        cdp_delta,
        cdp_eps,
        cdp_rho,
    )
except ImportError:
    # We allow import error for now so we can run the test file and see it fail on assertions or imports inside tests
    pass

def test_imports():
    """Test that we can import the DP module."""
    try:
        import sdeval.dp
    except ImportError:
        pytest.fail("Could not import sdeval.dp")

def test_gaussian_mechanism():
    """Test Gaussian mechanism basic functionality."""
    try:
        from sdeval.dp import gaussian_mechanism
    except ImportError:
        pytest.fail("Could not import gaussian_mechanism")

    # Test with scalar
    val = torch.tensor([100.0])
    sigma = 1.0
    noisy = gaussian_mechanism(val, sigma)
    assert isinstance(noisy, torch.Tensor)
    assert noisy.shape == val.shape
    assert noisy.item() != 100.0  # Extremely unlikely to be exactly equal

    # Test with tensor
    val = torch.zeros(1000)
    sigma = 10.0
    noisy = gaussian_mechanism(val, sigma)
    # Check that std dev is roughly sigma
    assert 9.0 < noisy.std().item() < 11.0

def test_laplace_mechanism():
    """Test Laplace mechanism basic functionality."""
    try:
        from sdeval.dp import laplace_mechanism
    except ImportError:
        pytest.fail("Could not import laplace_mechanism")

    val = torch.zeros(1000)
    scale = 10.0
    noisy = laplace_mechanism(val, scale)
    # Variance of Laplace(b) is 2b^2, so std is sqrt(2)*b
    expected_std = np.sqrt(2) * scale
    assert (expected_std - 2.0) < noisy.std().item() < (expected_std + 2.0)

def test_exponential_mechanism():
    """Test Exponential mechanism basic functionality."""
    try:
        from sdeval.dp import exponential_mechanism
    except ImportError:
        pytest.fail("Could not import exponential_mechanism")

    # Scores where one is clearly better
    scores = torch.tensor([1.0, 100.0, 1.0])
    epsilon = 10.0 # High epsilon -> should pick best
    sensitivity = 1.0
    
    idx = exponential_mechanism(scores, epsilon, sensitivity)
    assert idx.item() == 1

def test_privacy_conversions():
    """Test privacy parameter conversions."""
    try:
        from sdeval.dp import cdp_rho, cdp_delta
    except ImportError:
        pytest.fail("Could not import conversion functions")

    epsilon = 1.0
    delta = 1e-5
    rho = cdp_rho(epsilon, delta)
    assert rho > 0
    
    # Convert back
    calc_delta = cdp_delta(rho, epsilon)
    assert calc_delta <= delta
