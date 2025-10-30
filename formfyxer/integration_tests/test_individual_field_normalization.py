#!/usr/bin/env python3
"""
Integration test showing how the enhanced normalize_name function 
can be used for individual field normalization with PDF context.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from formfyxer.lit_explorer import normalize_name
from formfyxer.pdf_wrangling import get_existing_pdf_fields

def test_individual_field_normalization():
    """Test individual field normalization with PDF context."""
    
    pdf_path = "/home/quinten/FormFyxer/dev-testing/ML_training/auto/3a09aadafc4a4732b41ad48fd313ffe8.pdf"
    
    # Sample PDF context that might be extracted
    sample_context = """
    PETITION FOR ADOPTION OF STEPCHILD
    
    Petitioner's Name: ________________  Phone: ________________
    Address: ________________ City: ________________ State: __ Zip: _____
    
    Birth Parent's Name: ________________  Phone: ________________
    Address: ________________ City: ________________
    
    Child's Present Name: ________________
    Child's Name After Adoption: ________________
    Date of Birth: ____/____/____
    
    Adoptive Stepparent's Name: ________________
    Address: ________________ Phone: ________________
    
    Case Number: ________________
    Court: ________________ County: ________________
    """
    
    # Test with some fields that would benefit from context
    test_fields = [
        "pname",      # Should become petitioners_name or users1_name  
        "bpname",     # Should become users1_birth_parent_name
        "aspname",    # Should become users1_adoptive_stepparent_name
        "adoptname",  # Should become users1_adoptee_name_after_adoption
        "caseno",     # Should become case_number
        "county"      # Should become court_county
    ]
    
    print("Testing individual field normalization with context...")
    print("=" * 70)
    
    for i, field_name in enumerate(test_fields):
        print(f"\nField {i+1}: '{field_name}'")
        
        # Normalize with context
        try:
            normalized_name, confidence = normalize_name(
                jur="",                    # Legacy parameter
                group="adoption",          # Category hint
                n=i,                       # Position
                per=i/len(test_fields),    # Progress
                last_field="",             # Previous field
                this_field=field_name,     # Field to normalize
                context=sample_context,    # PDF context
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            print(f"  Normalized: '{normalized_name}' (confidence: {confidence:.2f})")
            
            if confidence > 0.7:
                print("  ✅ High confidence LLM normalization")
            elif confidence > 0.4:
                print("  ⚠️  Medium confidence normalization")
            else:
                print("  ❌ Low confidence, likely fallback used")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("Individual field normalization test completed!")

if __name__ == "__main__":
    test_individual_field_normalization()