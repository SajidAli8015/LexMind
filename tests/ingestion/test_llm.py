"""
Test suite for LLM client connectivity.
Tests all configured providers and verifies
the get_llm() function works correctly.

Run with: python -m tests.test_llm
      or: pytest tests/test_llm.py
"""

from loguru import logger
from src.llm_client import get_available_providers, test_llm_connection, get_llm


def test_available_providers():
    """Test that get_available_providers returns correct structure."""
    providers = get_available_providers()

    assert isinstance(providers, dict), "Should return a dict"

    required_keys = ["google", "openai", "anthropic", "azure"]
    for key in required_keys:
        assert key in providers, f"Missing provider: {key}"
        assert "configured" in providers[key], f"Missing 'configured' for {key}"
        assert "model" in providers[key], f"Missing 'model' for {key}"

    print("  test_available_providers PASSED")
    return True


def test_active_provider_connection():
    """Test connection to the currently configured provider."""
    result = test_llm_connection()
    assert result is True, "LLM connection test failed"
    print("  test_active_provider_connection PASSED")
    return True


def test_llm_returns_response():
    """Test that get_llm() returns a working LLM object."""
    llm = get_llm()
    response = llm.invoke("Reply with exactly: TEST_OK")
    assert response is not None, "Response should not be None"
    assert hasattr(response, "content"), "Response should have content"
    assert len(response.content) > 0, "Response content should not be empty"
    print(f"  test_llm_returns_response PASSED — response: {response.content[:50]}")
    return True


def run_all_tests():
    """Run all LLM client tests and report results."""
    print("=" * 50)
    print("LLM CLIENT TESTS")
    print("=" * 50)

    print("\nConfigured providers:")
    providers = get_available_providers()
    for name, info in providers.items():
        status = "READY" if info["configured"] else "NOT CONFIGURED"
        print(f"  {name}: {status} — {info['model']}")

    print()
    tests = [
        test_available_providers,
        test_active_provider_connection,
        test_llm_returns_response,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            print(f"Running {test_func.__name__}...")
            test_func()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
