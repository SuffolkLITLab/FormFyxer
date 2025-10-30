#!/usr/bin/env python3
"""
Integration test for refactored LLM-powered functions in FormFyxer.
Tests all functions that were changed to use external prompts and gpt-5-nano.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path (go up two levels from formfyxer/integration_tests/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from formfyxer.lit_explorer import (
    cluster_screens,
    plain_lang,
    guess_form_name,
    describe_form,
    _load_prompt
)

def test_prompt_loading():
    """Test that all external prompts load correctly."""
    print("=" * 60)
    print("TESTING PROMPT LOADING")
    print("=" * 60)
    
    prompts_to_test = [
        "field_grouping",
        "plain_language", 
        "guess_form_name",
        "describe_form"
    ]
    
    for prompt_name in prompts_to_test:
        try:
            prompt_content = _load_prompt(prompt_name)
            print(f"✅ {prompt_name}: {len(prompt_content)} characters loaded")
            print(f"   Preview: {prompt_content[:100]}...")
        except Exception as e:
            print(f"❌ {prompt_name}: Failed to load - {e}")
    
    print()

def test_plain_lang():
    """Test plain language rewriting function."""
    print("=" * 60)
    print("TESTING PLAIN_LANG FUNCTION")
    print("=" * 60)
    
    test_text = """The aforementioned party shall execute the requisite documentation 
    pursuant to the statutory requirements and in accordance with the established 
    jurisprudential precedents governing such contractual obligations."""
    
    print(f"Input text: {test_text}")
    print("\nCalling plain_lang()...")
    
    try:
        result = plain_lang(test_text)
        print(f"✅ Success! Result: {result}")
        
        # Basic validation - result should be different and simpler
        if len(result) > 0 and result != test_text:
            print("✅ Result is non-empty and different from input")
        else:
            print("⚠️  Result may not be properly processed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()

def test_guess_form_name():
    """Test form name guessing function."""
    print("=" * 60) 
    print("TESTING GUESS_FORM_NAME FUNCTION")
    print("=" * 60)
    
    # Sample court form text
    test_text = """NOTICE OF MOTION AND MOTION FOR SUMMARY JUDGMENT
    
    TO: All parties and their attorneys of record
    
    PLEASE TAKE NOTICE that on [DATE], at [TIME], in Department [X] of the above-entitled Court,
    plaintiff will move the Court for an order granting summary judgment in favor of plaintiff
    and against defendant on the grounds that there is no triable issue of material fact."""
    
    print(f"Input text: {test_text[:200]}...")
    print("\nCalling guess_form_name()...")
    
    try:
        result = guess_form_name(test_text)
        print(f"✅ Success! Form name: {result}")
        
        # Basic validation
        if len(result) > 0 and "abortthisnow" not in result.lower():
            print("✅ Result suggests this was identified as a court form")
        elif "abortthisnow" in result.lower():
            print("ℹ️  Result suggests this was not identified as a court form")
        else:
            print("⚠️  Unexpected result format")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()

def test_describe_form():
    """Test form description function."""
    print("=" * 60)
    print("TESTING DESCRIBE_FORM FUNCTION") 
    print("=" * 60)
    
    # Sample court form text
    test_text = """PETITION FOR DISSOLUTION OF MARRIAGE
    
    The petitioner respectfully requests that the Court grant a dissolution of marriage
    between the parties based on irreconcilable differences. The petitioner further
    requests that the Court make orders regarding property division, spousal support,
    and any other relief the Court deems just and proper."""
    
    print(f"Input text: {test_text[:200]}...")
    print("\nCalling describe_form()...")
    
    try:
        result = describe_form(test_text)
        print(f"✅ Success! Description: {result}")
        
        # Basic validation
        if len(result) > 0 and "abortthisnow" not in result.lower():
            print("✅ Result suggests this was identified as a court form")
        elif "abortthisnow" in result.lower():
            print("ℹ️  Result suggests this was not identified as a court form")
        else:
            print("⚠️  Unexpected result format")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()

def test_cluster_screens():
    """Test field clustering function."""
    print("=" * 60)
    print("TESTING CLUSTER_SCREENS FUNCTION")
    print("=" * 60)
    
    # Sample field names from a typical form
    test_fields = [
        "first_name",
        "last_name", 
        "date_of_birth",
        "social_security_number",
        "street_address",
        "city",
        "state", 
        "zip_code",
        "phone_number",
        "email_address",
        "employer_name",
        "job_title",
        "annual_income",
        "case_number",
        "court_name",
        "judge_name",
        "hearing_date"
    ]
    
    print(f"Input fields ({len(test_fields)} total):")
    for field in test_fields:
        print(f"  - {field}")
    
    print("\nCalling cluster_screens()...")
    
    try:
        result = cluster_screens(test_fields)
        print(f"✅ Success! Clustering result type: {type(result)}")
        
        if isinstance(result, dict):
            print(f"✅ Result is a dictionary with {len(result)} screens:")
            for screen_name, fields in result.items():
                print(f"  Screen '{screen_name}': {len(fields)} fields")
                for field in fields[:3]:  # Show first 3 fields
                    print(f"    - {field}")
                if len(fields) > 3:
                    print(f"    ... and {len(fields) - 3} more")
                    
            # Validate all fields are preserved
            all_result_fields = []
            for fields in result.values():
                all_result_fields.extend(fields)
            
            if set(all_result_fields) == set(test_fields):
                print("✅ All input fields preserved in output")
            else:
                print("⚠️  Some fields may be missing or duplicated")
                missing = set(test_fields) - set(all_result_fields)
                extra = set(all_result_fields) - set(test_fields)
                if missing:
                    print(f"   Missing: {missing}")
                if extra:
                    print(f"   Extra: {extra}")
        
        elif isinstance(result, str):
            print(f"✅ Result is a string (fallback mode): {result[:200]}...")
            
        else:
            print(f"⚠️  Unexpected result type: {type(result)}")
            print(f"   Result: {result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()

def main():
    """Run all integration tests."""
    print("FormFyxer LLM Functions Integration Test")
    print("Testing refactored functions with external prompts and gpt-5-nano")
    print("=" * 80)
    
    # Check for .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
    else:
        print("⚠️  .env file not found - API key may not be available")
    
    # Check for API key in environment
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OPENAI_API_KEY found in environment")
    else:
        print("⚠️  OPENAI_API_KEY not found in environment")
    
    print()
    
    # Run all tests
    test_prompt_loading()
    test_plain_lang()
    test_guess_form_name() 
    test_describe_form()
    test_cluster_screens()
    
    print("=" * 80)
    print("Integration test complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()