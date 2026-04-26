# Contributing to WAMP-proxy

First off, thank you for considering contributing to **Weighted Attention Message Pruner (WAMP)**! It's people like you that make the open-source community such an amazing place to learn, inspire, and create.

## 🛠️ Local Development Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/youruser/wamp-proxy.git
    cd wamp-proxy
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install ruff pytest  # For development
    ```

## 🧪 Running Tests

Always ensure your changes pass all tests before submitting a pull request.

```bash
# Run unit tests
export PYTHONPATH="."
pytest
```

## 🧹 Code Style

We use **Ruff** for linting and formatting. Please run it before committing:

```bash
# Check for linting issues
ruff check .

# Format the code
ruff format .
```

## 🚀 Pull Request Process

1.  Create a new branch for your feature or bugfix.
2.  Make your changes and add tests if necessary.
3.  Ensure all tests pass and the code is formatted.
4.  Submit a Pull Request with a clear description of what you've changed.

## 🛡️ Security

If you discover a security vulnerability, please do not open a public issue. Instead, refer to our `SECURITY.md`.

---
*By contributing, you agree that your contributions will be licensed under the project's MIT License.*
