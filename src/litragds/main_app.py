"""
Main application logic (separated from startup checks)
"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

def app_main(model_path_v):
    """Main application logic"""
    #setup_logging()

    # ... остальной код из предыдущего main.py ...
    # (интерактивный интерфейс, меню и т.д.)

    print("🚀 Application starting...")

    # Здесь будет основной код приложения
    # Пока просто тест
    
    print(f"cwd = {os.getcwd()}")
    print(f"   Model path: {model_path_v}")
    model = SentenceTransformer(model_path_v, local_files_only=True)

    test_text = ["Тестируем локальную модель"]
    embedding = model.encode(test_text)

    print(f"✅ Model working! Embedding shape: {embedding.shape}")


if __name__ == "__main__":
    model_path = "../models_cache/paraphrase-multilingual-MiniLM-L12-v2"
    app_main(model_path)