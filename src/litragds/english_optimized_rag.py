# src/english_optimized_rag.py
from litragds.optimized_rag import OptimizedRAG
from typing import Dict, Any
import os


class EnglishOptimizedRAG(OptimizedRAG):
    """Optimized RAG system for English literature with local model"""

    def __init__(self, deepseek_api_key: str = None, model_path: str = None):
        # Try to find English model
        if model_path is None:
            english_models = [
                "./models_cache/all-mpnet-base-v2",
                "./models_cache/all-MiniLM-L6-v2",
                "./models_cache/all-MiniLM-L4-v2"
            ]

            for model in english_models:
                if os.path.exists(model):
                    model_path = model
                    print(f"📦 Using English model: {model}")
                    break

            if model_path is None:
                model_path = "./models_cache/paraphrase-multilingual-MiniLM-L12-v2"
                print("⚠️ Using multilingual model for English")

        super().__init__(
            deepseek_api_key=deepseek_api_key,
            model_path=model_path
        )