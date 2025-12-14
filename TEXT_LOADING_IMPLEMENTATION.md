# Plain Text File Loading Implementation

## Overview
This implementation adds the ability to load plain text files in addition to FB2 files in the literature RAG system.

## Files Created

### 1. `/workspace/src/litragds/text_loader.py`
- New text loader class that handles plain text files
- Supports various text formats: `.txt`, `.text`, `.md`, `.markdown`
- Handles multiple encodings: utf-8, windows-1251, cp1251, koi8-r, iso-8859-1
- Automatically detects document structure (chapters, sections)
- Creates metadata for each text section

## Files Modified

### 1. `/workspace/src/litragds/optimized_rag.py`
- Added import for TextLoader
- Added text_loader instance variable
- Added `add_text_files()` method to load text files into the vector database
- Maintains consistency with existing `add_fb2_files()` method

### 2. `/workspace/src/litragds/load_books.py`
- Updated file detection to include text file extensions
- Added logic to process both FB2 and text files
- Added separate counters for FB2 and text file chunks
- Updated success message to reflect combined loading

## Features

### TextLoader Capabilities
- **File Format Support**: `.txt`, `.text`, `.md`, `.markdown` (case insensitive)
- **Encoding Support**: Automatically tries multiple encodings to handle different text files
- **Structure Detection**: Identifies chapters and sections using patterns like "Глава N", "Chapter N", etc.
- **Metadata Extraction**: Extracts basic metadata from filenames and content structure

### Integration Points
- **RAG Systems**: Both RussianOptimizedRAG and EnglishOptimizedRAG inherit the new functionality
- **Command Line Tool**: The load_books.py script now processes both FB2 and text files
- **Metadata Consistency**: Text files use the same metadata structure as FB2 files with 'source' field indicating 'text'

## Usage

### Programmatic Usage
```python
from litragds.russian_optimized_rag import RussianOptimizedRAG

rag_system = RussianOptimizedRAG(deepseek_api_key="your_key", model_path="path/to/model")
# Load text files
chunks_added = rag_system.add_text_files("/path/to/text/files")
```

### Command Line Usage
```bash
cd /workspace/src
python -c "from litragds.load_books import main_lb; main_lb(model_path_v='...', fb2_directory_v='/path/to/mixed/files', indices_directory_v='...', deepseek_api_key='...')"
```

The system will automatically detect and process both FB2 files (`.fb2`, `.zip`, `.fb2.zip`) and text files (`.txt`, `.text`, `.md`, `.markdown`).

## Benefits

1. **Extended Format Support**: Users can now load plain text files in addition to FB2 files
2. **Automatic Structure Detection**: The system intelligently identifies document sections
3. **Consistent Processing**: Text files are processed using the same chunking and embedding pipeline as FB2 files
4. **Metadata Preservation**: Text files get appropriate metadata tags for retrieval and analysis
5. **Backward Compatibility**: Existing FB2 functionality remains unchanged