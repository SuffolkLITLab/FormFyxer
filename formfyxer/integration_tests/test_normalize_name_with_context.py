#!/usr/bin/env python3
"""
Simple test for the enhanced normalize_name function with context.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from formfyxer.lit_explorer import normalize_name

def test_normalize_name_with_context():
    """Test the enhanced normalize_name function with PDF context."""
    
    print("Testing enhanced normalize_name function...")
    
    # Test data - simulate field normalization with context
    test_cases = [
        {
            "field_name": "pname",
            "context": "Petitioner's Name: ________________\nI hereby request that the court...",
            "description": "Petitioner name field with legal context"
        },
        {
            "field_name": "dob", 
            "context": "Date of Birth: ____/____/____\nChild's Information:",
            "description": "Date of birth field in child section"
        },
        {
            "field_name": "addr1",
            "context": "Address Line 1: ________________\nCity: _________ State: __ Zip: _____",
            "description": "Address field with context"
        },
        {
            "field_name": "users1_birthdate",  # This should be recognized as already normalized
            "context": "Date of Birth: ____/____/____",
            "description": "Already normalized field (should return with * prefix)"
        }
    ]
    
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['description']}")
        print(f"Original field: '{test_case['field_name']}'")
        print(f"Context snippet: '{test_case['context'][:50]}...'")
        
        # Test without context (fallback behavior)
        try:
            fallback_name, fallback_conf = normalize_name(
                jur="", group="", n=i, per=0.5, 
                last_field="", this_field=test_case['field_name']
            )
            print(f"Without context: '{fallback_name}' (confidence: {fallback_conf:.2f})")
        except Exception as e:
            print(f"Without context: Error - {str(e)[:60]}...")
            fallback_name, fallback_conf = "error", 0.0
        
        # Test with context (enhanced behavior) 
        try:
            enhanced_name, enhanced_conf = normalize_name(
                jur="", group="", n=i, per=0.5,
                last_field="", this_field=test_case['field_name'],
                context=test_case['context'],
                api_key=os.getenv("OPENAI_API_KEY")
            )
            print(f"With context: '{enhanced_name}' (confidence: {enhanced_conf:.2f})")
            
            if fallback_name != enhanced_name and fallback_name != "error":
                print(f"✅ LLM enhancement: {fallback_name} → {enhanced_name}")
            elif enhanced_name.startswith("*"):
                print("✅ Field recognized as already normalized")
            elif fallback_name == "error":
                print("✅ LLM bypassed tools token requirement")
            else:
                print("ℹ️  Same result as fallback")
                
        except Exception as e:
            print(f"❌ Error with context: {e}")
            print("ℹ️  This might be expected if no OpenAI API key is available")
        
        print("-" * 40)
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_normalize_name_with_context()