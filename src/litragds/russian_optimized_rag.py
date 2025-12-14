# src/russian_optimized_rag.py
from litragds.optimized_rag import OptimizedRAG
from typing import Dict, Any


class RussianOptimizedRAG(OptimizedRAG):
    """Optimized RAG system for Russian literature with local model"""

    def __init__(self, deepseek_api_key: str, model_path: str):

        super().__init__(
            deepseek_api_key=deepseek_api_key,
            model_path=model_path
        )

    def get_system_prompt(self) -> str:
        return """Ты - эксперт по русской литературе. Отвечай на вопросы на русском языке, 
используя предоставленный контекст из художественных произведений. Будь точным и информативным."""

    def get_chunker_settings(self) -> Dict[str, Any]:
        return {
            'optimal_chunk_size': 1200,
            'max_chunk_size': 2500,
            'min_chunk_size': 400,
            'overlap_size': 150,
            'max_total_chars': 50_000_000
        }