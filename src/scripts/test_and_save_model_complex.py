#!/usr/bin/env python3
import filter_stderror # noqa: F401
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
from pathlib import Path
from shutil import rmtree
import json
import os

model_path = "../models_cache/paraphrase-multilingual-MiniLM-L12-v2"
model_path_saved = "../models_cache/paraphrase-multilingual-MiniLM-L12-v2/saved"

dst_path = Path(model_path_saved)
if dst_path.exists():
    rmtree(dst_path)
dst_path.mkdir()

try:
    model = SentenceTransformer(model_path, local_files_only=True)
    print("✅ Original model loads successfully!")

except Exception as e:
    print(f"❌ Error loading original model: {e}")
    exit()

# Get the original transformer configuration
original_transformer = model[0]

# Create new transformer with the same configuration
transformer = Transformer(
    model_name_or_path=model_path,
    max_seq_length=original_transformer.max_seq_length,
    model_args={'local_files_only': True}
)

# Load the trained weights
transformer.auto_model.load_state_dict(original_transformer.auto_model.state_dict())
transformer.tokenizer = original_transformer.tokenizer

# Create pooling layer from original model
pooling_layer = model[1]
pooling_config = {
    'word_embedding_dimension': transformer.get_word_embedding_dimension(),
    'pooling_mode_cls_token': getattr(pooling_layer, 'pooling_mode_cls_token', False),
    'pooling_mode_mean_tokens': getattr(pooling_layer, 'pooling_mode_mean_tokens', True),
    'pooling_mode_max_tokens': getattr(pooling_layer, 'pooling_mode_max_tokens', False),
    'pooling_mode_mean_sqrt_len_tokens': getattr(pooling_layer, 'pooling_mode_mean_sqrt_len_tokens', False),
    'pooling_mode_weightedmean_tokens': getattr(pooling_layer, 'pooling_mode_weightedmean_tokens', False),
    'pooling_mode_lasttoken': getattr(pooling_layer, 'pooling_mode_lasttoken', False),
}

pooling = Pooling(**pooling_config)

# Save each module manually to ensure proper structure
print("Saving modules...")

# Save transformer module
transformer_path = os.path.join(model_path_saved, "0_Transformer")
os.makedirs(transformer_path, exist_ok=True)
transformer.save(transformer_path)

# Save pooling module
pooling_path = os.path.join(model_path_saved, "1_Pooling")
os.makedirs(pooling_path, exist_ok=True)
pooling.save(pooling_path)

# Create modules.json
modules_config = [
    {
        "idx": 0,
        "name": "Transformer",
        "path": "0_Transformer",
        "type": "sentence_transformers.models.Transformer"
    },
    {
        "idx": 1,
        "name": "Pooling",
        "path": "1_Pooling",
        "type": "sentence_transformers.models.Pooling"
    }
]

with open(os.path.join(model_path_saved, "modules.json"), "w") as f:
    json.dump(modules_config, f, indent=2)

# Create config_sentence_transformers.json
config_st = {
    "max_seq_length": original_transformer.max_seq_length,
    "do_lower_case": False
}

with open(os.path.join(model_path_saved, "config_sentence_transformers.json"), "w") as f:
    json.dump(config_st, f, indent=2)

# Copy README.md if it exists
readme_src = os.path.join(model_path, "README.md")
readme_dst = os.path.join(model_path_saved, "README.md")
if os.path.exists(readme_src):
    import shutil
    shutil.copy2(readme_src, readme_dst)

print("✅ Model saved successfully!")

# Test loading the saved model
try:
    model_saved = SentenceTransformer(model_path_saved, local_files_only=True)
    print("✅ Saved model loads successfully!")
    
    # Verify embeddings match
    test_sentence = "This is a test sentence."
    original_embedding = model.encode(test_sentence)
    saved_embedding = model_saved.encode(test_sentence)
    
    import numpy as np
    similarity = np.dot(original_embedding, saved_embedding) / (
        np.linalg.norm(original_embedding) * np.linalg.norm(saved_embedding)
    )
    print(f"✅ Embedding similarity: {similarity:.6f}")
    
    # List saved directory structure
    print("\n📁 Saved directory structure:")
    for root, dirs, files in os.walk(model_path_saved):
        level = root.replace(model_path_saved, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')

except Exception as e:
    print(f"❌ Error loading saved model: {e}")
    import traceback
    traceback.print_exc()
