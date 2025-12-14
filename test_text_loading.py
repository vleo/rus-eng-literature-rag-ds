#!/usr/bin/env python3
"""
Test script to verify that plain text files can be loaded
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from litragds.text_loader import TextLoader

def test_text_loader():
    print("Testing TextLoader...")
    
    # Test single file parsing
    sample_file = "/workspace/test_text_files/sample.txt"
    if os.path.exists(sample_file):
        print(f"Parsing file: {sample_file}")
        result = TextLoader.parse_text_file(sample_file)
        
        print(f"Result keys: {list(result.keys())}")
        print(f"Metadata: {result.get('metadata', {})}")
        print(f"Number of content sections: {len(result.get('content', []))}")
        
        for i, section in enumerate(result.get('content', [])):
            print(f"Section {i}: '{section['title']}' ({len(section['text'])} chars)")
            print(f"  Sample: {section['text'][:100]}...")
            print()
        
        # Test directory loading
        print("\nTesting directory loading...")
        documents = TextLoader.load_text_directory("/workspace/test_text_files/")
        print(f"Loaded {len(documents)} documents from directory")
        
        for i, doc in enumerate(documents):
            print(f"Document {i}: {doc['metadata']['title']} - {len(doc['text'])} chars")
            print(f"  Metadata: {doc['metadata']}")
            print()
    else:
        print(f"Sample file does not exist: {sample_file}")

if __name__ == "__main__":
    test_text_loader()