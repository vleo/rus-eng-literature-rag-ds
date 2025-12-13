# !/usr/bin/env python3
# scripts/convert_transformers_to_sentence_transformers.py
"""
Convert Transformers model format to Sentence Transformers format
"""
import filter_stderror # noqa: F401

import json
import shutil
from pathlib import Path


def convert_model(source_path: str, target_path: str = None):
    """
    Convert Transformers format to Sentence Transformers format

    Args:
        source_path: Path to Transformers model (with 0_Transformer, 1_Pooling folders)
        target_path: Where to save Sentence Transformers format (optional)
    """
    sbert_config = None
    source_dir = Path(source_path)

    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_path}")
        return False

    # Check if it's Transformers format
    transformer_dir = source_dir / "0_Transformer"
    # pooling_dir = source_dir / "1_Pooling"

    if not transformer_dir.exists():
        print(f"❌ Not a Transformers format: {transformer_dir} not found")
        return False

    # Determine target directory
    if target_path is None:
        # Create a new directory with "_sentence" suffix
        target_dir = Path(str(source_dir) + "_sentence")
    else:
        target_dir = Path(target_path)

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Converting: {source_dir.name}")
    print(f"   From: {source_dir.absolute()}")
    print(f"   To:   {target_dir.absolute()}")

    # Step 1: Copy main model files from 0_Transformer
    print("\n📦 Copying model files...")

    files_to_copy = [
        ("config.json", "config.json"),
        ("model.safetensors","model.safetensors"),
        ("pytorch_model.bin", "pytorch_model.bin"),
        ("tokenizer_config.json", "tokenizer_config.json"),
        ("special_tokens_map.json", "special_tokens_map.json"),
    ]

    for src_name, dst_name in files_to_copy:
        src_file = transformer_dir / src_name
        dst_file = target_dir / dst_name

        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            print(f"  ✅ {src_name} -> {dst_name}")
        else:
            print(f"  ⚠️  {src_name}: not found")

    # Step 2: Handle vocabulary files
    vocab_files = [
        "vocab.txt",  # BERT tokenizer
        "sentencepiece.bpe.model",  # SentencePiece tokenizer
        "spiece.model",  # XLM-R tokenizer
        "tokenizer.json",  # Fast tokenizer
    ]

    for vocab_file in vocab_files:
        src_file = transformer_dir / vocab_file
        if src_file.exists():
            shutil.copy2(src_file, target_dir / vocab_file)
            print(f"  ✅ {vocab_file} (vocabulary)")
            break

    # Step 3: Create sentence_bert_config.json
    print("\n⚙️ Creating sentence_bert_config.json...")

    # Read original config
    config_path = transformer_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Create Sentence BERT config
        sbert_config = {
            "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence_embedding_dimension": config.get("hidden_size", 384),
            "max_seq_length": config.get("max_position_embeddings", 512),
            "do_lower_case": False,
            "pooling_mode_cls_token": False,
            "pooling_mode_mean_tokens": True,
            "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False,
        }

        # Save sentence_bert_config.json
        sbert_config_path = target_dir / "sentence_bert_config.json"
        with open(sbert_config_path, 'w') as f:
            json.dump(sbert_config, f, indent=2)

        print(f"  ✅ Created sentence_bert_config.json")
        print(f"     Dimension: {sbert_config['sentence_embedding_dimension']}")
        print(f"     Max sequence: {sbert_config['max_seq_length']}")

    # Step 4: Create README.md
    readme_content = f"""# {target_dir.name}

Sentence Transformers model converted from Transformers format.

Original model: {source_dir.name}
Converted on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Details
- **Model Type:** Sentence Transformer
- **Base Model:** paraphrase-multilingual-MiniLM-L12-v2
- **Dimensions:** {sbert_config.get('sentence_embedding_dimension', 384)}
- **Max Sequence Length:** {sbert_config.get('max_seq_length', 512)}

## Usage
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('{target_dir.absolute()}', local_files_only=True)
embeddings = model.encode(["Your text here"])
```
"""

    with open(target_dir / "README.md", 'w') as f:
        f.write(readme_content)

    print(f"\n✅ Conversion complete!")
    print(f"📁 Target directory: {target_dir.absolute()}")

    # Test the converted model
    print("\n🧪 Testing converted model...")
    test_converted_model(str(target_dir))

    return True


def test_converted_model(model_path: str):
    """Test if converted model works"""
    try:
        from sentence_transformers import SentenceTransformer

        print(f"  Loading: {model_path}")
        model = SentenceTransformer(model_path, local_files_only=True)

        # Test encoding
        test_texts = ["Hello world", "Привет мир"]
        embeddings = model.encode(test_texts)

        print(f"  ✅ Model loaded successfully!")
        print(f"  📊 Embedding shape: {embeddings.shape}")
        print(f"  📊 Dimension: {embeddings.shape[1]}")

        return True

    except Exception as e1:
        print(f"  ❌ Error testing model: {e1}")
        return False


def main():
    """Main function"""
    print("🔄 Transformers to Sentence Transformers Converter")
    print("=" * 60)

    # Check current structure
    models_cache = Path("../models_cache")

    if not models_cache.exists():
        print(f"❌ models_cache directory not found")
        return

    # Find Transformers format models
    transformers_models = []

    for item in models_cache.iterdir():
        if item.is_dir():
            transformer_dir = item / "0_Transformer"
            if transformer_dir.exists():
                transformers_models.append(item)

    if not transformers_models:
        print("❌ No Transformers format models found")
        print("\n💡 Your current structure:")
        print_tree(models_cache)
        return

    print(f"🔍 Found {len(transformers_models)} Transformers format model(s):")
    for i, model in enumerate(transformers_models, 1):
        print(f"  {i}. {model.name}")

    print("\nSelect model to convert:")
    print("  a. Convert all models")
    print("  n. Convert specific model")

    choice = input("\nYour choice: ").strip().lower()

    if choice == 'a':
        # Convert all models
        for model in transformers_models:
            print(f"\n{'=' * 60}")
            convert_model(str(model))
    elif choice.isdigit():
        # Convert specific model
        idx = int(choice) - 1
        if 0 <= idx < len(transformers_models):
            convert_model(str(transformers_models[idx]))
        else:
            print("❌ Invalid selection")
    else:
        # Default: convert first model
        convert_model(str(transformers_models[0]))


def print_tree(directory: Path, prefix: str = ""):
    """Print directory tree"""
    contents = list(directory.iterdir())

    for i, item in enumerate(contents):
        is_last = (i == len(contents) - 1)

        if item.is_dir():
            print(f"{prefix}{'└── ' if is_last else '├── '}{item.name}/")
            extension = "    " if is_last else "│   "
            print_tree(item, prefix + extension)
        else:
            print(f"{prefix}{'└── ' if is_last else '├── '}{item.name}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Conversion interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")