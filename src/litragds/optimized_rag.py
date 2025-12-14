# src/optimized_rag.py
from litragds.base_rag import BaseRAGSystem
from litragds.fb2_loader import FB2Loader
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class OptimizedRAG(BaseRAGSystem):
    """Optimized RAG system with chunking and FB2 support using local model"""

    def __init__(self, deepseek_api_key: str, model_path: str):
        super().__init__(
            deepseek_api_key=deepseek_api_key,
            model_path=model_path,
            local_files_only=True
        )
        self.fb2_loader = FB2Loader()
        self.total_chars_processed = 0

        # Test model
        if not self.test_model():
            raise RuntimeError("Model failed to load or test")

    def add_documents_optimized(self, texts: List[str], metadata: List[Dict] = None) -> int:
        """Add documents with language-optimized chunking"""
        if metadata is None:
            metadata = [{}] * len(texts)

        settings = self.get_chunker_settings()
        chunk_size = settings['optimal_chunk_size']
        overlap = settings['overlap_size']

        all_chunks = []
        all_metadata = []

        print(f"📊 Processing {len(texts)} texts with chunk size {chunk_size}...")

        for text, meta in zip(texts, metadata):
            # Check total volume
            if self.total_chars_processed + len(text) > settings['max_total_chars']:
                logger.warning("Exceeded maximum database size. Text not added.")
                continue

            # Split text into chunks
            chunks = self._chunk_text(text, chunk_size, overlap)

            for i, chunk in enumerate(chunks):
                # Skip too short chunks
                if len(chunk) < settings['min_chunk_size']:
                    continue

                chunk_meta = meta.copy()
                chunk_meta.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size_chars': len(chunk),
                    'is_chunk': True
                })

                all_chunks.append(chunk)
                all_metadata.append(chunk_meta)

            self.total_chars_processed += len(text)

        # Add to vector database
        if all_chunks:
            print(f"📥 Adding {len(all_chunks)} chunks to database...")
            self.add_documents(all_chunks, all_metadata)

        logger.info(f"Added {len(all_chunks)} chunks from {len(texts)} texts")
        return len(all_chunks)

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Chunk text using language-appropriate boundaries"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            if end < len(text):
                # Find optimal boundary
                boundary_pos = self._find_optimal_boundary(text, end)
                if boundary_pos > start:
                    end = boundary_pos
                else:
                    # Find space boundary
                    space_pos = text.rfind(' ', start, end)
                    if space_pos != -1 and space_pos > start + chunk_size // 2:
                        end = space_pos

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start with overlap
            start = end - overlap if end - overlap > start else end

            if start >= len(text):
                break

        return chunks

    def _find_optimal_boundary(self, text: str, position: int) -> int:
        """Find optimal break point"""
        boundaries = ['\n\n', '\n', '. ', '! ', '? ', '; ']

        for boundary in boundaries:
            search_start = max(0, position - 200)
            search_end = min(len(text), position + 200)

            pos = text.rfind(boundary, search_start, search_end)
            if pos != -1:
                return pos + len(boundary)

        return position

    def add_fb2_files(self, directory_path: str, file_pattern: str = "*.fb2") -> int:
        """Add FB2 files to vector database"""
        print(f"📚 Loading FB2 files from {directory_path}...")
        documents = self.fb2_loader.load_fb2_directory(directory_path, file_pattern)

        if not documents:
            logger.warning("No documents extracted from FB2 files")
            return 0

        texts = [doc['text'] for doc in documents]
        metadata = [doc['metadata'] for doc in documents]

        return self.add_documents_optimized(texts, metadata)

    def get_system_prompt(self) -> str:
        """Return generic system prompt for multi-language literature"""
        return """You are an expert in literature analysis. 
Answer questions using the provided context from literary works. 
Be accurate, informative and cite sources when possible."""

    def get_chunker_settings(self) -> Dict[str, Any]:
        """Return generic chunking settings for multi-language support"""
        return {
            'optimal_chunk_size': 1000,
            'max_chunk_size': 2000,
            'min_chunk_size': 300,
            'overlap_size': 100,
            'max_total_chars': 50_000_000
        }

    def get_library_stats(self) -> Dict[str, Any]:
        """Get library statistics"""
        stats = {
            'total_documents': len(self.documents),
            'total_authors': set(),
            'total_works': set(),
            'genres': set(),
            'languages': set()
        }

        for meta in self.metadata:
            if 'author' in meta and meta['author']:
                stats['total_authors'].add(meta['author'])
            if 'title' in meta and meta['title']:
                stats['total_works'].add(meta['title'])
            if 'genre' in meta and meta['genre']:
                stats['genres'].add(meta['genre'])
            if 'language' in meta and meta['language']:
                stats['languages'].add(meta['language'])

        # Convert sets to lists
        stats['total_authors'] = list(stats['total_authors'])
        stats['total_works'] = list(stats['total_works'])
        stats['genres'] = list(stats['genres'])
        stats['languages'] = list(stats['languages'])

        return stats