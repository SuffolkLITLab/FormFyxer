#!/usr/bin/env python3
"""Test script to verify API key resolution works correctly."""

import os
import sys
from pathlib import Path

# Add the formfyxer package to the path (go up two levels from integration_tests to reach the root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from formfyxer.docassemble_support import (
    get_openai_api_key,
    get_openai_api_key_from_sources,
    is_docassemble_available,
)


def test_api_key_resolution():
    """Test the API key resolution functions."""
    print("🔬 Testing API key resolution functions")
    print("=" * 50)

    # Test 1: Environment variable (should work)
    print("\n1. Testing environment variable resolution:")
    env_key = get_openai_api_key()
    if env_key:
        print(f"✅ Found API key from environment: {env_key[:10]}...")
    else:
        print("❌ No API key found in environment")

    # Test 2: Explicit key takes precedence
    print("\n2. Testing explicit key takes precedence:")
    explicit_key = "test-explicit-key"
    result = get_openai_api_key(explicit_key)
    if result == explicit_key:
        print("✅ Explicit key correctly takes precedence")
    else:
        print(f"❌ Expected explicit key, got: {result}")

    # Test 3: Test sources function
    print("\n3. Testing get_openai_api_key_from_sources:")
    creds = {"key": "test-creds-key", "org": "test-org"}

    # Explicit key should take precedence over creds
    result = get_openai_api_key_from_sources("explicit-key", creds)
    if result == "explicit-key":
        print("✅ Explicit key takes precedence over creds")
    else:
        print(f"❌ Expected explicit key, got: {result}")

    # Creds key should be used if no explicit key
    result = get_openai_api_key_from_sources(None, creds)
    if result == "test-creds-key":
        print("✅ Creds key used when no explicit key")
    else:
        print(f"❌ Expected creds key, got: {result}")

    # Test 4: Test docassemble availability and fallback
    print("\n4. Testing docassemble config detection:")
    docassemble_available = is_docassemble_available()
    print(f"💡 Docassemble available: {docassemble_available}")

    result = get_openai_api_key()  # Should fall back to environment
    if result:
        print(f"✅ API key resolution works: {result[:10]}...")
        if docassemble_available:
            print("   (May include docassemble config)")
        else:
            print("   (Environment fallback)")
    else:
        print("❌ No API key found from any source")

    print("\n" + "=" * 50)
    print("🏁 API key resolution test completed!")


def test_integration():
    """Test that the integration works end-to-end."""
    print("\n🔬 Testing end-to-end integration")
    print("=" * 50)
    print("⚠️  Note: This may take time due to NLTK/docassemble initialization")

    try:
        print("   Importing passive voice detection module...")
        from formfyxer.passive_voice_detection import detect_passive_voice_segments

        # Load API key from .env file if available
        env_file = Path(__file__).parent.parent.parent / ".env"
        api_key = None
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("OPENAI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        os.environ["OPENAI_API_KEY"] = api_key
                        break

        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")

        # Test with explicit API key
        if api_key and api_key.startswith("sk-"):
            print("\n1. Testing passive voice detection with explicit API key:")
            result = detect_passive_voice_segments(
                "The ball was thrown by John.", api_key=api_key
            )
            print(f"✅ Explicit API key parameter works: {len(result)} results")

            print("\n2. Testing passive voice detection with default resolution:")
            result = detect_passive_voice_segments("The ball was thrown by John.")
            print(f"✅ Default API key resolution works: {len(result)} results")
        else:
            print("⚠️  No valid OPENAI_API_KEY found, skipping live tests")
            print("   (API key should start with 'sk-' to be valid)")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")


if __name__ == "__main__":
    import sys

    test_api_key_resolution()

    # Only run integration test if not skipped
    if "--skip-integration" not in sys.argv:
        test_integration()
    else:
        print("\n⚠️  Integration test skipped (use without --skip-integration to run)")
        print(
            "   Integration test imports full modules which may be slow due to NLTK/docassemble"
        )
