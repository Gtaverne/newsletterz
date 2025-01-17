# Newsletterz

A Python application to fetch, index, and search through your Gmail newsletters using semantic search and natural language queries.

## Features
Once you fetched emails, the program can run entirely locally, on a laptop, without external connexions.

## Overview

This application consists of two main flows:

1. **Email Processing Pipeline**: 
   - Authenticates with Gmail
   - Fetches newsletters
   - Extracts and cleans text content
   - Generates embeddings using Ollama
   - Stores in ChromaDB for semantic search

2. **Search Interface**:
   - Natural language query understanding
   - Semantic search in the email database
   - Smart response generation
   - Interactive CLI interface

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Ollama
- Google Cloud Platform account
- Tested on MacOS only

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/gtaverne/newsletterz
   cd newsletterz
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start ChromaDB**
   ```bash
   cd docker
   docker-compose up -d
   ```

4. **Install Ollama and required models**
   ```bash
   # Install Ollama from https://ollama.ai
   
   # Pull required models
   ollama pull llama3
   ollama pull qwen2.5-coder:32b
   ollama pull mxbai-embed-large
   ```

## Configuration

1. **Set up Google Cloud Platform**
   - Create a new project in GCP Console
   - Enable the Gmail API
   - Create OAuth 2.0 credentials
   - Download the credentials JSON file
   - Place it in `secrets/credentials.json`

2. **Environment Variables**
   ```bash
   cp .env.template .env
   # Edit .env with your configuration
   ```

## Usage

1. **First-time setup: Fetch and index emails**
   ```bash
   python -m src.email.email_processor
   ```
   This will:
   - Authenticate with Gmail
   - Fetch your newsletters
   - Process and store them in ChromaDB

2. **Search your emails**
   ```bash
   python -m src.interface.dialog_interface
   ```
   Example queries:
   - "What are the latest AI trends from McKinsey?"
   - "Show me cloud computing articles from big tech companies"
   - "Summarize what consulting firms say about digital transformation"

## Architecture

The application is structured into several key components:

- `src/email/`: Email fetching and processing
- `src/search/`: Search and query processing
- `src/interface/`: CLI interface
- `tests/`: Test suites
- `docker/`: Docker configuration for ChromaDB

## Development

1. **Running tests**
   ```bash
   pytest tests/
   ```

2. **Code style**
   ```bash
   black .
   flake8
   ```

## Common Issues

1. **ChromaDB Connection**
   - Ensure Docker is running
   - Check if ChromaDB container is healthy
   - Default port is 8183

2. **Gmail Authentication**
   - First-time auth requires browser access
   - Token is stored locally for future use
   - Check credentials.json path

3. **Ollama**
   - Ensure Ollama service is running
   - Check model downloads
   - Default port is 11434

## License

MIT