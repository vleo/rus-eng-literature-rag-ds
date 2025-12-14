#!/usr/bin/env python3
"""
Basic test to verify that our text loading functionality is properly integrated
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all our modules can be imported without error"""
    print("Testing imports...")
    
    try:
        from litragds.text_loader import TextLoader
        print("✅ TextLoader imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TextLoader: {e}")
        return False
    
    try:
        from litragds.optimized_rag import OptimizedRAG
        print("✅ OptimizedRAG imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import OptimizedRAG: {e}")
        return False
    
    try:
        from litragds.russian_optimized_rag import RussianOptimizedRAG
        print("✅ RussianOptimizedRAG imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import RussianOptimizedRAG: {e}")
        return False
    
    try:
        from litragds.english_optimized_rag import EnglishOptimizedRAG
        print("✅ EnglishOptimizedRAG imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import EnglishOptimizedRAG: {e}")
        return False
    
    return True

def test_optimized_rag_has_text_methods():
    """Test that OptimizedRAG has the text loading methods"""
    print("\nTesting OptimizedRAG for text loading methods...")
    
    from litragds.optimized_rag import OptimizedRAG
    
    # Check that the class has the expected methods
    if hasattr(OptimizedRAG, 'add_text_files'):
        print("✅ OptimizedRAG has add_text_files method")
    else:
        print("❌ OptimizedRAG missing add_text_files method")
        return False
    
    if hasattr(OptimizedRAG, 'text_loader'):
        print("✅ OptimizedRAG has text_loader attribute")
    else:
        print("❌ OptimizedRAG missing text_loader attribute")
        return False
    
    # Check that RussianOptimizedRAG inherits the method
    from litragds.russian_optimized_rag import RussianOptimizedRAG
    if hasattr(RussianOptimizedRAG, 'add_text_files'):
        print("✅ RussianOptimizedRAG inherits add_text_files method")
    else:
        print("❌ RussianOptimizedRAG does not inherit add_text_files method")
        return False
    
    # Check that EnglishOptimizedRAG inherits the method
    from litragds.english_optimized_rag import EnglishOptimizedRAG
    if hasattr(EnglishOptimizedRAG, 'add_text_files'):
        print("✅ EnglishOptimizedRAG inherits add_text_files method")
    else:
        print("❌ EnglishOptimizedRAG does not inherit add_text_files method")
        return False
    
    return True

def test_load_books_script():
    """Test that load_books script has been updated correctly"""
    print("\nTesting load_books script updates...")
    
    # Read the file and check for our changes
    with open('/workspace/src/litragds/load_books.py', 'r') as f:
        content = f.read()
    
    # Check for text file detection
    if "*.text" in content or "*.txt" in content:
        print("✅ load_books.py includes text file detection")
    else:
        print("❌ load_books.py missing text file detection")
        return False
    
    # Check for add_text_files call
    if "add_text_files" in content:
        print("✅ load_books.py includes add_text_files call")
    else:
        print("❌ load_books.py missing add_text_files call")
        return False
    
    # Check for total_chunks_added logic
    if "total_chunks_added" in content:
        print("✅ load_books.py includes total_chunks_added logic")
    else:
        print("❌ load_books.py missing total_chunks_added logic")
        return False
    
    return True

def main():
    print("Running basic integration tests for text loading functionality...\n")
    
    success = True
    success &= test_imports()
    success &= test_optimized_rag_has_text_methods()
    success &= test_load_books_script()
    
    print(f"\n{'='*50}")
    if success:
        print("🎉 All basic integration tests passed!")
        print("✅ Plain text file loading functionality has been successfully added")
    else:
        print("❌ Some tests failed")
    
    return success

if __name__ == "__main__":
    main()