# src/base_rag.py
import os
import json
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseRAGSystem(ABC):
    """Abstract base class for RAG systems"""

    DEFAULT_MODEL_PATH = "./models_cache/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, deepseek_api_key: str = None, model_path: str = None,
                 local_files_only: bool = True):
        """
        Args:
            model_path: Path to local model directory
            local_files_only: If True, only use local files, don't download
        """
        self.deepseek_api_key = deepseek_api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.deepseek_api_key:
            raise ValueError("DeepSeek API key not provided. Set DEEPSEEK_API_KEY in .env file")

        # Determine model path
        self.model_path = model_path or self.DEFAULT_MODEL_PATH

        # Check if model exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                f"Download model or check path. Expected files: "
                f"pytorch_model.bin, config.json, sentence_bert_config.json"
            )

        # Initialize embedding model from local files
        print(f"📦 Loading local model from: {self.model_path}")
        self.embedding_model = SentenceTransformer(
            self.model_path,
            local_files_only=local_files_only
        )

        # Get model dimensions
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded. Embedding dimension: {self.dimension}")

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)

        # Storage for texts
        self.documents = []
        self.metadata = []

        # DeepSeek API
        self.api_url = "https://api.deepseek.com/v1/chat/completions"

        logger.info(f"{self.__class__.__name__} initialized with local model")

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return system prompt for the specific language/literature"""
        pass

    @abstractmethod
    def get_chunker_settings(self) -> Dict[str, Any]:
        """Return chunking settings optimized for the language"""
        pass

    # ===== COMMON METHODS =====

    def add_documents(self, texts: List[str], metadata: List[Dict] = None):
        """Add documents to vector database"""
        if metadata is None:
            metadata = [{}] * len(texts)

        if len(texts) != len(metadata):
            raise ValueError("Texts and metadata must have the same length")

        # Generate embeddings
        print(f"📊 Generating embeddings for {len(texts)} documents...")
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        print(f"✅ Embeddings generated: {embeddings.shape}")

        # Add to FAISS index
        if self.index.ntotal == 0:
            self.index.add(embeddings.astype(np.float32))
        else:
            self.index.add(embeddings.astype(np.float32))

        # Store documents and metadata
        self.documents.extend(texts)
        self.metadata.extend(metadata)

        logger.info(f"Added {len(texts)} documents to vector database")

    def search_similar(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        query_embedding = query_embedding.astype(np.float32)

        # Search in FAISS
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    'text': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(score)
                })

        return results

    def query_deepseek(self, prompt: str, context: str = None, temperature: float = 0.7) -> str:
        """Query DeepSeek API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.deepseek_api_key}"
        }

        system_message = self.get_system_prompt()

        if context:
            user_content = f"Context:\n{context}\n\nQuestion: {prompt}"
        else:
            user_content = prompt

        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ],
            "temperature": temperature,
            "max_tokens": 2000
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content']

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return f"Error accessing API: {e}"

    def rag_query(self, question: str, k: int = 5) -> str:
        """Full RAG pipeline"""
        similar_docs = self.search_similar(question, k)

        if not similar_docs:
            return self.get_no_results_message()

        # Format context
        context = "\n\n".join([
            f"Excerpt {i + 1} (similarity: {doc['score']:.3f}):\n{doc['text']}"
            for i, doc in enumerate(similar_docs)
        ])

        response = self.query_deepseek(question, context)
        return response

    def get_no_results_message(self) -> str:
        """Return message when no results found"""
        return "No relevant information found for your question in the knowledge base."

    # ===== SAVE/LOAD METHODS =====

    def save_index(self, filepath: str):
        """Save vector index and documents"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")

        # Save documents and metadata
        data = {
            'documents': self.documents,
            'metadata': self.metadata,
            'model_path': self.model_path
        }

        with open(f"{filepath}.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Index saved to {filepath}")

    def load_index(self, filepath: str):
        """Load vector index and documents"""
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.index")

        # Load documents and metadata
        with open(f"{filepath}.json", 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.documents = data['documents']
        self.metadata = data['metadata']

        # Check if model path matches
        saved_model_path = data.get('model_path')
        if saved_model_path and saved_model_path != self.model_path:
            logger.warning(f"Model path mismatch. Saved: {saved_model_path}, Current: {self.model_path}")

        logger.info(f"Index loaded from {filepath}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        total_chars = sum(len(doc) for doc in self.documents)
        avg_chunk_size = total_chars / len(self.documents) if self.documents else 0

        settings = self.get_chunker_settings()

        return {
            'total_documents': len(self.documents),
            'total_characters': total_chars,
            'average_chunk_size': round(avg_chunk_size, 1),
            'memory_usage_percentage': round((total_chars / settings['max_total_chars']) * 100, 2),
            'recommended_max_chars': settings['max_total_chars'],
            'embedding_dimension': self.dimension,
            'model_path': self.model_path
        }

    def test_model(self) -> bool:
        """Test if model is working correctly"""
        try:
            test_text = ["Test sentence for model verification"]
            embedding = self.embedding_model.encode(test_text)
            print(f"✅ Model test passed. Embedding shape: {embedding.shape}")
            return True
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            return False