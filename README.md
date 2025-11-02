# RAG-Chatbot

A small Retrieval-Augmented Generation (RAG) chatbot that lets a user chat with the contents of their PDF documents. The project ingests PDF files, builds a vector store of embeddings, and answers user questions by retrieving relevant passages and generating concise responses.

This repository contains a minimal Python implementation intended for local use and experimentation. It is suitable as a starting point for building document-aware chat assistants.

## Features

- Ingest multiple PDFs and create a searchable vector index
- Retrieve relevant passages for a user question
- Generate answers using a language model with retrieved context
- Lightweight, focused codebase for learning and extension

## Project layout

- `main.py` — application entrypoint / example runner
- `src/herlper.py` — helper utilities for ingestion and query handling
- `src/prompt.py` — prompt templates and prompt-related helpers
- `trials.ipynb` — interactive notebook for experimenting
- `pyproject.toml` — Python packaging and dependency metadata

## Quick start

Prerequisites

- Python 3.10+ recommended
- A virtual environment (venv, conda, etc.)
- (Optional) An API key for the language model provider you plan to use

Install

1. Create and activate a virtual environment:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install project dependencies:

```powershell
pip install -r requirements.txt
```

If `requirements.txt` is not present, use `pyproject.toml` tooling (for example `pip install .` or `pip install -e .`).

Running the app

- There is a minimal `main.py` which demonstrates ingesting documents and starting a simple query loop. Run it with:

```powershell
python main.py
```

Check `trials.ipynb` for an interactive walkthrough which demonstrates how to ingest PDFs and run queries.

Configuration

- The project expects any model provider configuration (API keys, endpoints) to be provided via environment variables or a configuration file. Check `src/herlper.py` and `src/prompt.py` for places that load configuration.

Usage example

1. Place your PDF files in a folder the app can read.
2. Use the ingestion helper to index the files.
3. Ask natural-language questions — the system will retrieve supporting passages and generate an answer.

Development

- Run unit tests (if present) with your preferred test runner (pytest recommended):

```powershell
pytest -q
```

- Linting and formatting: use `ruff`/`black` or other tools as configured in the repository.

Troubleshooting

- "(" was not closed: If you encounter a Python SyntaxError complaining that a parenthesis was not closed, open `src/herlper.py` and look for any unterminated function calls or f-strings. A recent fix replaced a broken call with a proper handler and added a guard for a missing conversational chain.
- Missing dependencies: ensure you have installed the dependencies listed in `pyproject.toml` or `requirements.txt`.

Contributing

Contributions are welcome. Open an issue or submit a pull request with a clear description of the change.

License

This project is provided under the MIT License (see `LICENSE` file if present). If no license file exists, treat the repository as unlicensed and request clarification from the owner before using it in production.

Contact

For questions about this repository, open an issue or contact the repository owner.

