#!/usr/bin/env python3
"""
Main application file with local model support
"""

from litragds.filter_stderror import setup_logging
logger = setup_logging()

import os, sys
from pathlib import Path
from dotenv import load_dotenv

import litragds.main_app
import litragds.load_books

load_dotenv()

model_path_p = Path("./models_cache/paraphrase-multilingual-MiniLM-L12-v2")
fb2_directory = "./data/fb2_books"
indices_directory = "./indices"
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")


def check_prerequisites():
    """Check if all prerequisites are met"""
    print("🔍 Checking prerequisites...")

    # Check .env file
    if not os.path.exists(".env"):
        print("❌ .env file not found")
        print("   Copy .env.example to .env and add your DeepSeek API key")
        return False

    # Check model
    if not model_path_p.exists():
        print("❌ Model not found")
        print(f"   Expected at: {model_path_p.absolute()}")
        print("\n💡 Download the model:")
        print("   python scripts/download_models.py")
        return False

    # Check required files
    required_files = ["pytorch_model.bin", "config.json"]
    missing_files = []
    for file in required_files:
        if not (model_path_p / file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing model files: {', '.join(missing_files)}")
        return False

    print("✅ All prerequisites met!")
    return True


def run_simple_test():
    """Run the simple embedding test in a separate process"""
    try:
        # Run the simple test functionality as a subprocess
        # cmd = [
        #     sys.executable, "-c",
        #     f"import sys; sys.path.insert(0, './src'); "
        #     f"from litragds.main_app import app_main; "
        #     f"app_main('{str(model_path_p)}')"
        # ]
        # result = subprocess.run(cmd, cwd="/workspace", capture_output=False, text=True, check=True)
        
        litragds.main_app.app_main(model_path_v=f"{str(model_path_p)}")
        return True
    # except subprocess.CalledProcessError as e:
    #     print(f"❌ Error running simple test: {e}")
    #     return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def run_load_books():
    """Run the load_books functionality in a separate process"""
    try:
        litragds.load_books.main_lb(
            model_path_v=f"{str(model_path_p)}",
            fb2_directory_v=fb2_directory,
            indices_directory_v=indices_directory,
            deepseek_api_key=deepseek_api_key,
        )
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def show_menu():
    """Show menu options to user"""
    print("\n🎭 Literature RAG System - Menu")
    print("=" * 40)
    print("1. Test embedding functionality (simple test)")
    print("2. Load books into RAG system")
    print("3. Exit")
    print("-" * 40)


def main():
    """Main application"""
#    logger.info("🎭 Literature RAG System")
    print("\n🎭 Literature RAG System")
    print("=" * 50)

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Show menu and get user choice
    while True:
        show_menu()
        try:
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == "1":
                print("\n🏃 Running simple embedding test...")
                run_simple_test()
            elif choice == "2":
                print("\n📚 Running load books functionality...")
                run_load_books()
            elif choice == "3":
                print("\n👋 Goodbye!")
                break
            else:
                print("\n❌ Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            continue


if __name__ == "__main__":
    main()