# AnkiVec - Vector Search for Anki

Anki extension that creates vector embeddings for cards using Ollama and enables hybrid semantic search with ChromaDB.

## Features

- **Vector Embeddings**: Generate embeddings for all cards using local Ollama models
- **Semantic Search**: Find cards by meaning, not just keywords
- **Fast Local Processing**: Uses lightweight embedding models (nomic-embed-text by default)
- **Persistent Storage**: ChromaDB stores embeddings for quick retrieval

## Prerequisites

1. **Anki** (version 2.1.45+)
2. **Ollama** installed and running locally
3. The **uv+** package manager

### Install Ollama

Download from [ollama.ai](https://ollama.ai) and install.

Pull the embedding model:
```bash
ollama pull nomic-embed-text
```
