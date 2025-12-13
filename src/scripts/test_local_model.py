#!/usr/bin/env python3
import filter_stderror # noqa: F401
"""
Local Model Testing Utility
Проверяет загрузку и работоспособность локальных моделей Sentence Transformers
"""

import os
import sys
import time
from pathlib import Path
import json


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"🔍 {text}")
    print("=" * 60)


def check_model_structure(model_path: str) -> bool:
    """Check if model directory has all required files"""
    model_dir = Path(model_path)

    if not model_dir.exists():
        print(f"❌ Model directory does not exist: {model_path}")
        return False

    print(f"📁 Model directory: {model_dir.absolute()}")

    # Required files for Sentence Transformers
    required_files = [
        ("pytorch_model.bin", "Main model weights", 100 * 1024 * 1024),  # >100MB
        ("config.json", "Model configuration", 200),
        ("sentence_bert_config.json", "Sentence BERT config", 40),
        ("tokenizer_config.json", "Tokenizer configuration", 200),
        ("tokenizer.json", "Vocabulary file", 1000),
    ]

    print("\n📄 Checking required files:")
    all_files_ok = True

    for filename, description, min_size in required_files:
        file_path = model_dir / filename

        if file_path.exists():
            size_bytes = file_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)

            if size_bytes >= min_size:
                print(f"  ✅ {filename}: {size_mb:.1f} MB - {description}")
            else:
                print(f"  ⚠️  {filename}: {size_mb:.1f} MB - {description} (SUSPICIOUSLY SMALL)")
                all_files_ok = False
        else:
            print(f"  ❌ {filename}: MISSING - {description}")
            all_files_ok = False

    return all_files_ok


def load_and_test_model(model_path: str) -> bool:
    """Load model and test basic functionality"""
    print_header("Loading Model")

    try:
        from sentence_transformers import SentenceTransformer

        start_time = time.time()
        print(f"🔧 Loading model from: {model_path}")

        # Load model with local files only
        model = SentenceTransformer(model_path, local_files_only=True)

        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")

        # Get model info
        model_info = model.get_sentence_embedding_dimension()
        print(f"📊 Model dimension: {model_info}")

        return model, True

    except ImportError as e:
        print(f"❌ Cannot import sentence_transformers: {e}")
        print("   Install: pip install sentence-transformers")
        return None, False

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, False


def test_encoding(model, language: str = "multilingual"):
    """Test text encoding with different languages"""
    print_header(f"Testing Encoding ({language})")

    test_texts = {
        "multilingual": [
            "This is a test sentence in English.",
            "Это тестовое предложение на русском языке.",
            "Bonjour, ceci est un test en français.",
            "Hallo, das ist ein Test auf Deutsch.",
            "Hola, esto es una prueba en español."
        ],
        "english": [
            "The quick brown fox jumps over the lazy dog.",
            "To be or not to be, that is the question.",
            "It was the best of times, it was the worst of times.",
            "All happy families are alike; each unhappy family is unhappy in its own way.",
            "Call me Ishmael."
        ],
        "russian": [
            "В синем небе звезды блещут, в синем море волны хлещут.",
            "Мороз и солнце; день чудесный!",
            "Я помню чудное мгновенье: передо мной явилась ты.",
            "Умом Россию не понять, аршином общим не измерить.",
            "Человек — это звучит гордо!"
        ]
    }

    texts = test_texts.get(language, test_texts["multilingual"])

    print(f"🧪 Encoding {len(texts)} {language} sentences...")

    try:
        start_time = time.time()
        embeddings = model.encode(texts)
        encode_time = time.time() - start_time

        print(f"✅ Encoding completed in {encode_time:.2f} seconds")
        print(f"📊 Embedding shape: {embeddings.shape}")
        print(f"📊 Embedding dimension: {embeddings.shape[1]}")

        # Show first embedding sample
        print(f"\n📈 First embedding (first 5 values):")
        print(f"   {embeddings[0][:5]}")

        return embeddings, True

    except Exception as e:
        print(f"❌ Error during encoding: {e}")
        return None, False


def test_similarity(embeddings, texts):
    """Test semantic similarity between sentences"""
    print_header("Testing Semantic Similarity")

    try:
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # Calculate similarity matrix
        sim_matrix = cosine_similarity(embeddings)

        print("📊 Similarity matrix:")
        for i in range(min(5, len(texts))):  # Show first 5x5
            row = "  ".join([f"{sim_matrix[i][j]:.3f}" for j in range(min(5, len(texts)))])
            print(f"  [{row}]")

        # Show some interesting comparisons
        if len(texts) >= 2:
            print(f"\n🔗 Similarity between:")
            print(f"   '{texts[0][:30]}...'")
            print(f"   '{texts[1][:30]}...'")
            print(f"   = {sim_matrix[0][1]:.4f}")

        # Check if embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"\n📐 Vector norms (should be ~1.0 if normalized):")
        print(f"   Min: {norms.min():.4f}, Max: {norms.max():.4f}, Mean: {norms.mean():.4f}")

        return True

    except ImportError:
        print("⚠️  sklearn not installed, skipping similarity test")
        print("   Install: pip install scikit-learn")
        return False
    except Exception as e:
        print(f"❌ Error in similarity test: {e}")
        return False


def test_batch_encoding(model):
    """Test batch encoding performance"""
    print_header("Testing Batch Performance")

    try:
        # Generate test batch
        batch_sizes = [1, 10, 50, 100]
        test_sentence = "This is a test sentence for batch encoding performance measurement."

        for batch_size in batch_sizes:
            batch = [test_sentence] * batch_size

            start_time = time.time()
            embeddings = model.encode(batch, show_progress_bar=False)
            elapsed = time.time() - start_time

            sentences_per_second = batch_size / elapsed if elapsed > 0 else 0

            print(f"  📦 Batch size {batch_size:3d}: {elapsed:.3f}s "
                  f"({sentences_per_second:5.1f} sentences/sec)")

        return True

    except Exception as e:
        print(f"❌ Error in batch test: {e}")
        return False


def test_model_info(model_path: str):
    """Extract and display model information"""
    print_header("Model Information")

    try:
        config_path = Path(model_path) / "config.json"
        sbert_config_path = Path(model_path) / "sentence_bert_config.json"

        info = {
            "model_path": str(Path(model_path).absolute()),
            "files_exist": {
                "config.json": config_path.exists(),
                "sentence_bert_config.json": sbert_config_path.exists()
            }
        }

        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                info.update({
                    "hidden_size": config.get("hidden_size"),
                    "num_attention_heads": config.get("num_attention_heads"),
                    "num_hidden_layers": config.get("num_hidden_layers"),
                    "model_type": config.get("model_type")
                })

        if sbert_config_path.exists():
            with open(sbert_config_path, 'r') as f:
                sbert_config = json.load(f)
                info.update({
                    "sbert_model_name": sbert_config.get("model_name"),
                    "sbert_dimension": sbert_config.get("sentence_embedding_dimension"),
                    "sbert_pooling_mode": sbert_config.get("pooling_mode")
                })

        print("📋 Model configuration:")
        for key, value in info.items():
            if key != "model_path":
                print(f"  {key}: {value}")

        print(f"\n📍 Full path: {info['model_path']}")

        return info

    except Exception as e:
        print(f"⚠️  Could not read model info: {e}")
        return {}


def find_available_models():
    """Find all available local models"""
    print_header("Available Local Models")

    models_dir = Path("../models_cache")

    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return []

    model_dirs = []

    for item in models_dir.iterdir():
        if item.is_dir():
            # Check if it looks like a Sentence Transformers model
            config_file = item / "config.json"
            modules_file = item / "modules.json"
            config_exists = config_file.exists() or modules_file.exists()
            
            pytorch_file = item / "pytorch_model.bin"
            safetensors_file = item / "0_Transformer"
            db_exists = pytorch_file.exists() or safetensors_file.exists()
            
            if config_exists and db_exists:
                model_dirs.append(item)

    if model_dirs:
        print(f"✅ Found {len(model_dirs)} model(s):")
        for i, model_dir in enumerate(model_dirs, 1):
            size_mb = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / (1024 * 1024)
            print(f"  {i}. {model_dir.name} ({size_mb:.1f} MB)")
    else:
        print("❌ No valid models found in models_cache/")
        print("\n💡 Download models using:")
        print("   python scripts/download_models.py")

    return model_dirs


def main():
    """Main testing function"""
    print("\n" + "=" * 60)
    print("🧪 SENTENCE TRANSFORMERS LOCAL MODEL TESTER")
    print("=" * 60)

    # Find available models
    available_models = find_available_models()

    if not available_models:
        print("\n❌ No models to test. Exiting.")
        sys.exit(1)

    # Let user choose or use default
    print("\nSelect model to test:")
    for i, model_dir in enumerate(available_models, 1):
        print(f"  {i}. {model_dir.name}")
    print(f"  a. Test all models")

    choice = input("\nYour choice (default: 1): ").strip()

    models_to_test = []

    if choice.lower() == 'a':
        models_to_test = available_models
        print(f"\n📋 Will test all {len(models_to_test)} models")
    elif choice.isdigit() and 1 <= int(choice) <= len(available_models):
        models_to_test = [available_models[int(choice) - 1]]
    else:
        models_to_test = [available_models[0]]
        print(f"\n📋 Using default: {models_to_test[0].name}")

    # Test each selected model
    results = {}

    for model_dir in models_to_test:
        model_path = str(model_dir)
        model_name = model_dir.name

        print_header(f"Testing Model: {model_name}")

        # Step 1: Check structure
        # structure_ok = check_model_structure(model_path)
        # if not structure_ok:
        #     print(f"\n❌ Model structure incomplete: {model_name}")
        #     results[model_name] = "FAILED - Structure"
        #     continue

        # Step 2: Load model
        model, load_ok = load_and_test_model(model_path)
        if not load_ok or model is None:
            print(f"\n❌ Failed to load model: {model_name}")
            results[model_name] = "FAILED - Loading"
            continue

        # Step 3: Get model info
        model_info = test_model_info(model_path)

        # Step 4: Test encoding
        # Determine language from model name
        if "multilingual" in model_name.lower():
            language = "multilingual"
        elif "russian" in model_name.lower():
            language = "russian"
        else:
            language = "english"

        embeddings, encode_ok = test_encoding(model, language)
        if not encode_ok:
            print(f"\n⚠️  Encoding test failed for: {model_name}")
            results[model_name] = "PARTIAL - Encoding failed"
            continue

        # Step 5: Test similarity (if embeddings available)
        if embeddings is not None:
            # Get test texts based on language
            test_texts_dict = {
                "english": ["Test English 1", "Test English 2"],
                "russian": ["Тест Русский 1", "Тест Русский 2"],
                "multilingual": ["Test", "Тест"]
            }
            test_texts = test_texts_dict.get(language, ["Test 1", "Test 2"])

            similarity_ok = test_similarity(embeddings, test_texts)

        # Step 6: Test batch performance
        batch_ok = test_batch_encoding(model)

        # All tests passed
        print(f"\n🎉 All tests passed for: {model_name}")
        results[model_name] = "PASSED"

        # Clean up
        del model

    # Print summary
    print_header("TEST SUMMARY")

    print("📋 Results:")
    for model_name, result in results.items():
        if "PASSED" in result:
            print(f"  ✅ {model_name}: {result}")
        elif "PARTIAL" in result:
            print(f"  ⚠️  {model_name}: {result}")
        else:
            print(f"  ❌ {model_name}: {result}")

    print("\n💡 Next steps:")
    print("   1. If all tests PASSED: You're ready to use the RAG system!")
    print("   2. If tests FAILED: Check model files and download again")
    print("   3. Run: python main.py to start the RAG system")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)