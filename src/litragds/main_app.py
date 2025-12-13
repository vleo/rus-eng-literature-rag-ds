#!/usr/bin/env python3
#import litragds.filter_stderror # noqa: F401
"""
Main application logic (separated from startup checks)
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()


def setup_logging():
    logging.basicConfig(
        level=os.getenv('LOG_LEVEL', 'INFO'),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./logs/rag_system.log'),
            logging.StreamHandler()
        ]
    )
    # Reduce logging
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def app_main():
    """Main application logic"""
    setup_logging()

    # ... остальной код из предыдущего main.py ...
    # (интерактивный интерфейс, меню и т.д.)

    print("🚀 Application starting...")

    # Здесь будет основной код приложения
    # Пока просто тест
    from sentence_transformers import SentenceTransformer

    model_path = "./models_cache/paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_path, local_files_only=True)

    test_text = ["Тестируем локальную модель"]
    embedding = model.encode(test_text)

    print(f"✅ Model working! Embedding shape: {embedding.shape}")
    print(f"   Model path: {model_path}")


if __name__ == "__main__":
    app_main()