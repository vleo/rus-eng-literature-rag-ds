# lit_rag_ds

A lit RAG data science project

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for data science applications. It provides tools for semantic search, document analysis, and AI-powered insights extraction.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Install in development mode
```bash
pip install -e .
```

### Install with development dependencies
```bash
pip install -e ".[dev]"
```

## Usage

### Running the main application
```bash
python litragdstop/src/main.py
```

## Project Structure

```
lit_rag_ds/
├── pyproject.toml          # Project configuration and dependencies
├── README.md               # Documentation
├── LICENSE                 # License information
├── litragdstop/
│   └── src/
│       ├── main.py         # Main entry point
│       └── litragds/       # Core package modules
│           ├── __init__.py
│           └── example_module.py
└── tests/                  # Test files (optional)
```

## Development

### Setting up the development environment
1. Clone the repository
2. Install in development mode: `pip install -e .`
3. Install development dependencies: `pip install -e ".[dev]"`

### Running tests
```bash
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
