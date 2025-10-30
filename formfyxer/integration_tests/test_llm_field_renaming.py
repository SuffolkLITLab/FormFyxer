#!/usr/bin/env python3
"""
Integration test for LLM-powered field renaming functionality in FormFyxer.
Tests the complete workflow of loading a PDF, analyzing fields with full context,
using LLM to generate semantic field names, and saving the renamed PDF.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path so we can import formfyxer
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from formfyxer.pdf_wrangling import get_existing_pdf_fields, rename_pdf_fields
from formfyxer.lit_explorer import rename_pdf_fields_with_context
import shutil

def extract_field_names_from_pdf(pdf_path: str) -> list:
    """Extract field names from a PDF file."""
    try:
        fields_in_pages = get_existing_pdf_fields(pdf_path)
        all_field_names = []
        
        for page_fields in fields_in_pages:
            for field in page_fields:
                if hasattr(field, 'name') and field.name:
                    all_field_names.append(field.name)
        
        return list(set(all_field_names))  # Remove duplicates
    except Exception as e:
        print(f"Error extracting field names: {e}")
        return []

def main():
    # File paths
    input_pdf = "/home/quinten/FormFyxer/dev-testing/ML_training/auto/3a09aadafc4a4732b41ad48fd313ffe8.pdf"
    output_pdf = "/home/quinten/FormFyxer/test-renamed.pdf"
    
    print("=" * 60)
    print("FormFyxer LLM Field Renaming Test")
    print("=" * 60)
    
    # Check if input file exists
    if not os.path.exists(input_pdf):
        print(f"❌ Error: Input PDF not found at {input_pdf}")
        return
    
    print(f"📄 Input PDF: {input_pdf}")
    print(f"📄 Output PDF: {output_pdf}")
    print()
    
    # Step 1: Extract existing field names from the PDF (before processing)
    print("Step 1: Extracting original field names from PDF...")
    original_field_names = extract_field_names_from_pdf(input_pdf)
    
    if not original_field_names:
        print("❌ No fields found in the PDF or error occurred")
        return
    
    print(f"✅ Found {len(original_field_names)} original fields:")
    for i, name in enumerate(original_field_names, 1):
        print(f"  {i}. {name}")
    print()
    

    
    # Step 2: Use LLM to rename fields with full PDF context
    print("Step 2: Using LLM to rename fields with full PDF context...")
    try:
        field_mapping = rename_pdf_fields_with_context(
            pdf_path=input_pdf,
            original_field_names=original_field_names,
            api_key=os.getenv("OPENAI_API_KEY"),  # Use env var for API key
            model="gpt-4o-mini"  # Using a reliable model
        )
        
        if not field_mapping:
            print("❌ No field mappings returned from LLM")
            return
            
        print(f"✅ LLM generated {len(field_mapping)} field mappings:")
        for original, renamed in field_mapping.items():
            if original != renamed:
                print(f"  ✏️  {original} → {renamed}")
            else:
                print(f"  ➡️  {original} (unchanged)")
        print()
        
    except Exception as e:
        print(f"❌ Error during LLM field renaming: {e}")
        return
    
    # Step 3: Apply the field renaming to the PDF and save
    print("Step 3: Applying field renaming and saving PDF...")
    try:
        rename_pdf_fields(
            in_file=input_pdf,
            out_file=output_pdf,
            mapping=field_mapping
        )
        
        print(f"✅ Successfully saved renamed PDF to: {output_pdf}")
        
        # Verify the output file was created
        if os.path.exists(output_pdf):
            file_size = os.path.getsize(output_pdf)
            print(f"✅ Output file size: {file_size:,} bytes")
        else:
            print("❌ Output file was not created")
            return
        
    except Exception as e:
        print(f"❌ Error saving renamed PDF: {e}")
        return
    
    # Step 4: Verify the renamed fields in the output PDF
    print("\nStep 4: Verifying renamed fields in output PDF...")
    try:
        output_field_names = extract_field_names_from_pdf(output_pdf)
        print(f"✅ Output PDF contains {len(output_field_names)} fields:")
        for i, name in enumerate(output_field_names, 1):
            print(f"  {i}. {name}")
        
        # Show the field renaming changes
        print("\nField Renaming Summary:")
        print("=" * 40)
        if len(original_field_names) == len(output_field_names):
            for i, (orig, new) in enumerate(zip(original_field_names, output_field_names)):
                if orig != new:
                    print(f"  ✏️  {orig} → {new}")
                else:
                    print(f"  ➡️  {orig} (unchanged)")
        else:
            print(f"  ⚠️  Field count changed: {len(original_field_names)} → {len(output_field_names)}")
        
        print("\n" + "=" * 60)
        print("🎉 END-TO-END TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error verifying output PDF: {e}")

if __name__ == "__main__":
    main()