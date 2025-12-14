#!/usr/bin/env python3
"""
Test script to verify that plain text files can be loaded via RAG system
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from litragds.russian_optimized_rag import RussianOptimizedRAG

def test_rag_text_loading():
    print("Testing RAG system with text files...")
    
    try:
        # Initialize RAG system (this requires the model to be present)
        # We'll use a mock key and path - might fail if model isn't available
        model_path = "../models_cache/paraphrase-multilingual-MiniLM-L12-v2"
        
        # Adjust path to match the expected location relative to src
        full_model_path = "/workspace/src/" + model_path
        if not os.path.exists(full_model_path):
            print(f"Model not found at {full_model_path}, trying alternative...")
            # Look for any model in models_cache
            models_dir = "/workspace/src/models_cache"
            if os.path.exists(models_dir):
                subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
                if subdirs:
                    model_path = f"../models_cache/{subdirs[0]}"
                    print(f"Using model: {model_path}")
                else:
                    print("No models found in ../models_cache")
                    return
        
        print(f"Initializing RAG system with model: {model_path}")
        rag_system = RussianOptimizedRAG(deepseek_api_key="test_key", model_path=model_path)
        print("RAG system initialized successfully!")
        
        # Test loading text files
        text_dir = "/workspace/test_text_files"
        chunks_added = rag_system.add_text_files(text_dir)
        
        print(f"Added {chunks_added} text chunks to RAG system")
        
        # Print memory stats
        stats = rag_system.get_memory_stats()
        print(f"Memory stats: {stats}")
        
        # Print library stats
        lib_stats = rag_system.get_library_stats()
        print(f"Library stats: {lib_stats}")
        
    except Exception as e:
        print(f"Error during RAG text loading test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_text_loading()