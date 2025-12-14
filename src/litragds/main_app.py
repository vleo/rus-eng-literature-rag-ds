"""
Main application logic (separated from startup checks)
Interactive RAG system interface
"""

import os, sys
from dotenv import load_dotenv
from litragds.optimized_rag import OptimizedRAG
from litragds.russian_optimized_rag import RussianOptimizedRAG
from litragds.english_optimized_rag import EnglishOptimizedRAG
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def app_main(model_path_v):
    """Main application logic with interactive RAG querying"""
    
    print("🚀 Application starting...")
    print(f"cwd = {os.getcwd()}")
    print(f"Model path: {model_path_v}")

    # Ask for language preference
    print("\n🌍 Select language for RAG system:")
    print("1 - Russian / Русский")
    print("2 - English / Английский")
    print("3 - Generic Multilingual")
    
    lang_choice = input("\nYour choice: ").strip()
    
    # Initialize appropriate RAG system
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
    
    if lang_choice == '1':
        rag_system = RussianOptimizedRAG(deepseek_api_key=deepseek_api_key, model_path=model_path_v)
        print("✅ Russian RAG system initialized")
    elif lang_choice == '2':
        rag_system = EnglishOptimizedRAG(deepseek_api_key=deepseek_api_key, model_path=model_path_v)
        print("✅ English RAG system initialized")
    else:
        # Create a generic OptimizedRAG instance
        rag_system = OptimizedRAG(deepseek_api_key=deepseek_api_key, model_path=model_path_v)
        print("✅ Generic RAG system initialized")
    
    # Check if index exists and load it
    index_path = "../indices/literature_index"
    try:
        rag_system.load_index(index_path)
        print(f"✅ Vector index loaded from {index_path}")
        
        # Show statistics
        stats = rag_system.get_memory_stats()
        print(f"📊 Database stats: {stats['total_documents']} documents, {stats['total_characters']:,} chars")
        
        library_stats = rag_system.get_library_stats()
        print(f"📚 Library stats: {len(library_stats['total_authors'])} authors, {len(library_stats['total_works'])} works")
        
    except FileNotFoundError:
        print(f"⚠️ Index not found at {index_path}, starting with empty database")
        print("💡 Run load_books.py first to populate the database")
    except Exception as e:
        print(f"⚠️ Error loading index: {e}")
        print("Starting with empty database")

    # Interactive loop
    print("\n🎯 RAG System Ready!")
    print("Commands:")
    print("  - Type a question to get RAG response")
    print("  - 'test_embed' - Test embedding functionality")
    print("  - 'test_api' - Test DeepSeek API connection")
    print("  - 'stats' - Show database statistics")
    print("  - 'quit' or 'exit' - Exit the program")
    
    while True:
        try:
            user_input = input("\n❓ Enter your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'test_embed':
                # Test embedding functionality
                test_text = ["Test embedding functionality"]
                embedding = rag_system.embedding_model.encode(test_text)
                print(f"✅ Embedding test successful! Shape: {embedding.shape}")
            elif user_input.lower() == 'test_api':
                # Test API connection
                response = rag_system.query_deepseek("Hello, are you working?", temperature=0.1)
                print(f"🤖 API Response: {response}")
            elif user_input.lower() == 'stats':
                # Show statistics
                stats = rag_system.get_memory_stats()
                print(f"📊 Memory stats: {stats}")
                
                library_stats = rag_system.get_library_stats()
                print(f"📚 Library stats: {library_stats}")
            elif user_input:
                # Use RAG to answer the question
                print("🔍 Searching for relevant information...")
                response = rag_system.rag_query(user_input, k=5)
                print(f"🤖 Answer: {response}")
            else:
                print("Please enter a valid command or question.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing request: {e}")


if __name__ == "__main__":
    model_path = "../models_cache/paraphrase-multilingual-MiniLM-L12-v2"
    app_main(model_path)