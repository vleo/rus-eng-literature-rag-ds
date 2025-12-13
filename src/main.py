#!/usr/bin/env python3
"""
Main application file with local model support
"""
from litragds.filter_stderror import setup_logging
logger = setup_logging()

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from litragds.main_app import app_main

load_dotenv()

model_path_p = Path("./models_cache/paraphrase-multilingual-MiniLM-L12-v2")


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


def main():
    """Main application"""
    logger.info("🎭 Literature RAG System")
    print("\n🎭 Literature RAG System")
    print("=" * 50)

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Import after checks

    # Run the application
    app_main(str(model_path_p))


if __name__ == "__main__":
    main()