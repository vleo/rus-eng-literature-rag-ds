# src/load_books.py
# !/usr/bin/env python3
"""
Утилита для загрузки FB2 книг в RAG систему с локальной моделью
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from litragds.english_optimized_rag import EnglishOptimizedRAG
from litragds.russian_optimized_rag import RussianOptimizedRAG

# Add to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def check_model_exists(model_path_v: str = None) -> bool:
    """Check if model exists locally"""

    required_files = [
        "pytorch_model.bin",
        "config.json",
        "sentence_bert_config.json"
    ]

    model_dir = Path(model_path_v)
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_path_v}")
        return False

    missing_files = []
    for file in required_files:
        file_path = model_dir / file
        if not file_path.exists():
            missing_files.append(file)
        elif file_path.stat().st_size < 30:
            print(f"⚠️  File seems empty: {file}")

    if missing_files:
        print(f"❌ Missing model files: {', '.join(missing_files)}")
        print(f"📁 Model directory: {model_dir.absolute()}")
        print("\n💡 Download the model manually:")
        print("1. mkdir -p models_cache")
        print("2. cd models_cache")
        print("3. git lfs install")
        print("4. git clone https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        return False

    print(f"✅ Model found at: {model_path_v}")
    return True


def main_lb(model_path_v, fb2_directory_v, indices_directory_v, deepseek_api_key):
    """Основная функция загрузки книг"""
    load_dotenv()

    print("📚 Loading FB2 books into RAG system")
    print("=" * 40)

    # Check model
    if not check_model_exists(model_path_v=model_path_v):
        return

    # Check books directory
    if not os.path.exists(fb2_directory_v):
        print(f"❌ Directory {fb2_directory_v} not found")
        print("Create directory and add FB2 files")
        return

    # List FB2 files
    import glob
    fb2_files = glob.glob(os.path.join(fb2_directory_v, "*.fb2"))
    fb2_files.extend(glob.glob(os.path.join(fb2_directory_v, "*.fb2.zip")))
    fb2_files.extend(glob.glob(os.path.join(fb2_directory_v, "*.zip")))

    if not fb2_files:
        print(f"❌ No FB2 files found in {fb2_directory_v}")
        print("Add files with extensions: .fb2, .zip, .fb2.zip")
        return

    print(f"📁 Found {len(fb2_files)} FB2 files:")
    for file in fb2_files[:10]:  # Show first 10
        print(f"   📄 {os.path.basename(file)}")
    if len(fb2_files) > 10:
        print(f"   ... and {len(fb2_files) - 10} more")

    # Ask for language
    print("\n🌍 Select language for processing:")
    print("1 - Russian / Русский")
    print("2 - English / Английский")

    lang_choice = input("\nYour choice: ").strip()
    language = 'russian' if lang_choice == '1' else 'english'

    # Initialize RAG system
    try:
        if language == 'english':
            rag_system = EnglishOptimizedRAG()
            print("✅ English RAG system initialized")
        else:
            rag_system = RussianOptimizedRAG(deepseek_api_key=deepseek_api_key,model_path=model_path_v)
            print("✅ Russian RAG system initialized")
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return

    # Load books
    print(f"\n🔄 Loading books from {fb2_directory_v}...")
    chunks_added = rag_system.add_fb2_files(fb2_directory_v)

    if chunks_added > 0:
        # Save index
        index_path = f"{indices_directory_v}/literature_index_{language}"
        rag_system.save_index(index_path)

        # Show statistics
        stats = rag_system.get_memory_stats()
        library_stats = rag_system.get_library_stats()

        print(f"\n🎉 Loading completed!")
        print(f"✅ Added chunks: {stats['total_documents']}")
        print(f"✅ Processed characters: {stats['total_characters']:,}")
        print(f"✅ Authors: {len(library_stats['total_authors'])}")
        print(f"✅ Works: {len(library_stats['total_works'])}")
        print(f"💾 Index saved to: {index_path}")

        if library_stats['total_authors']:
            print(f"\n👥 Loaded authors: {', '.join(library_stats['total_authors'])}")

    else:
        print("❌ Failed to load books")


if __name__ == "__main__":
    model_path = "../models_cache/paraphrase-multilingual-MiniLM-L12-v2"
    fb2_directory = "../data/fb2_books"
    indices_directory = "../indices"
    
    main_lb(model_path_v=model_path, fb2_directory_v=fb2_directory, deepseek_api_key="xyzzy", indices_directory_v=indices_directory)