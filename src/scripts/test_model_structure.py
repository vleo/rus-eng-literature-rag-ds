# !/usr/bin/env python3
import filter_stderror # noqa: F401
import os
from sentence_transformers import SentenceTransformer

print(f"cwd = {os.getcwd()}")

model_path = "../models_cache/paraphrase-multilingual-MiniLM-L12-v2"
#model_path = "../models_cache/saved_sentence"

required_files = [
#    "pytorch_model.bin",
    "config.json",
    "sentence_bert_config.json",  # Самый важный!
    "tokenizer_config.json",
    "tokenizer.json"
]

print("🔍 Checking model structure...")
required_files_missing = False
for file in required_files:
    full_path = os.path.join(model_path, file)
    if os.path.exists(full_path):
        print(f"✅ {file}")
    else:
        required_files_missing = True
        print(f"❌ {file} - MISSING!")

if required_files_missing:
    print(f"❌ FILE(S) MISSING! Exiting...")
    exit(1)

# Try to load
print("\n🧪 Testing model load...")
try:
    model = SentenceTransformer(model_path, local_files_only=True)
    print("✅ Model loads successfully!")

except Exception as e:
    print(f"❌ Error: {e}")
