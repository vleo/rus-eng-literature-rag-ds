# !/usr/bin/env python3
import filter_stderror # noqa: F401 (redefined sys.stderr)

from sentence_transformers import SentenceTransformer

model_path = "../models_cache/paraphrase-multilingual-MiniLM-L12-v2"
# Test
model = SentenceTransformer( model_path, local_files_only=True )
assert model[1].pooling_mode_mean_tokens, "Expecting mean pooling"
print("✅ Loaded — spurious warnings silenced, real errors preserved.")