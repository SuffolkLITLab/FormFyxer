#!/usr/bin/env python3
"""Test script for passive voice detection module.

This script tests the passive_voice_detection module to ensure it works correctly
with the simplified OpenAI responses API approach.
"""

import sys
from pathlib import Path

# Add the formfyxer package to the path so we can import it
sys.path.insert(0, str(Path(__file__).parent))

# Import the specific module directly to avoid cv2 import issues
import importlib.util


def import_passive_voice_module():
    """Import the passive voice detection module directly."""
    module_path = Path(__file__).parent / "formfyxer" / "passive_voice_detection.py"
    spec = importlib.util.spec_from_file_location(
        "passive_voice_detection", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


passive_voice_module = import_passive_voice_module()
detect_passive_voice_segments = passive_voice_module.detect_passive_voice_segments


def test_basic_functionality():
    """Test basic passive voice detection with simple examples."""
    print("🔬 Testing basic functionality...")

    # Test clearly passive sentences
    test_cases = [
        ("The ball was thrown by John.", True),
        ("The president was impeached by Congress.", True),
        ("Letters were sent to all customers.", True),
        ("The door is being opened.", True),
        ("The work has been completed.", True),
        (
            "The politics being discussed were causing a scene.",
            True,
        ),  # Tricky case from our tests
    ]

    for sentence, should_be_passive in test_cases:
        try:
            result = detect_passive_voice_segments(sentence)

            if not result:
                print(f"❌ No result returned for: {sentence}")
                continue

            sentence_result, fragments = result[0]
            is_passive = len(fragments) > 0

            status = "✅" if is_passive == should_be_passive else "❌"
            voice_type = "passive" if is_passive else "active"
            expected_type = "passive" if should_be_passive else "active"

            print(f"{status} '{sentence}' → {voice_type} (expected {expected_type})")
            if fragments:
                print(f"   Fragments: {fragments}")

        except Exception as e:
            print(f"❌ Error processing '{sentence}': {e}")


def test_active_voice():
    """Test sentences that should be classified as active voice."""
    print("\n🔬 Testing active voice detection...")

    active_sentences = [
        "John threw the ball.",
        "The President gave a speech about COVID relief.",
        "I don't usually participate in discussions about politics.",
        "The world of politics is quite fascinating.",
        "Sanders fights for the working class.",
        "Politics is a topic for the commoners.",
    ]

    for sentence in active_sentences:
        try:
            result = detect_passive_voice_segments(sentence)

            if not result:
                print(f"❌ No result returned for: {sentence}")
                continue

            sentence_result, fragments = result[0]
            is_passive = len(fragments) > 0

            status = "✅" if not is_passive else "❌"
            voice_type = "passive" if is_passive else "active"

            print(f"{status} '{sentence}' → {voice_type} (expected active)")
            if fragments:
                print(f"   Unexpected fragments: {fragments}")

        except Exception as e:
            print(f"❌ Error processing '{sentence}': {e}")


def test_multiple_sentences():
    """Test processing multiple sentences at once."""
    print("\n🔬 Testing multiple sentence processing...")

    text = """
    The ball was thrown by John. 
    John threw the ball back. 
    The game was enjoyed by everyone.
    Everyone enjoyed the game.
    """

    try:
        results = detect_passive_voice_segments(text)

        print(f"Processed {len(results)} sentences:")
        for sentence, fragments in results:
            voice_type = "passive" if fragments else "active"
            print(f"  • '{sentence.strip()}' → {voice_type}")
            if fragments:
                print(f"    Fragments: {fragments}")

    except Exception as e:
        print(f"❌ Error processing multiple sentences: {e}")


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\n🔬 Testing edge cases...")

    # Test empty input
    try:
        result = detect_passive_voice_segments("")
        print("❌ Empty string should raise ValueError")
    except ValueError:
        print("✅ Empty string correctly raises ValueError")
    except Exception as e:
        print(f"❌ Unexpected error for empty string: {e}")

    # Test very short sentences (should be filtered out)
    try:
        result = detect_passive_voice_segments("Hi. Yes.")
        print("❌ Short sentences should raise ValueError")
    except ValueError:
        print("✅ Short sentences correctly raises ValueError")
    except Exception as e:
        print(f"❌ Unexpected error for short sentences: {e}")

    # Test list input
    try:
        sentences = [
            "The document was reviewed carefully.",
            "We reviewed the document carefully.",
        ]
        results = detect_passive_voice_segments(sentences)

        print(f"✅ List input processed {len(results)} sentences:")
        for sentence, fragments in results:
            voice_type = "passive" if fragments else "active"
            print(f"  • '{sentence}' → {voice_type}")

    except Exception as e:
        print(f"❌ Error processing list input: {e}")


def test_api_availability():
    """Test if the OpenAI API is properly configured."""
    print("\n🔬 Testing API availability...")

    try:
        # Try a simple test
        result = detect_passive_voice_segments("This is a simple test sentence.")

        if result:
            sentence, fragments = result[0]
            voice_type = "passive" if fragments else "active"
            print(
                f"✅ API connection successful! Test sentence classified as: {voice_type}"
            )
        else:
            print("❌ API returned empty result")

    except Exception as e:
        print(f"❌ API connection failed: {e}")
        print("💡 Make sure OPENAI_API_KEY is set in your environment or .env file")


def main():
    """Run all tests."""
    print("🧪 Testing FormFyxer Passive Voice Detection Module")
    print("=" * 55)

    test_api_availability()
    test_basic_functionality()
    test_active_voice()
    test_multiple_sentences()
    test_edge_cases()

    print("\n" + "=" * 55)
    print("🏁 Test suite completed!")
    print("\n💡 Tips:")
    print("   - If API tests fail, check your OPENAI_API_KEY environment variable")
    print("   - Compare results with the promptfoo evaluation for consistency")
    print("   - Check that the model (gpt-5-nano) is accessible with your API key")


if __name__ == "__main__":
    main()
