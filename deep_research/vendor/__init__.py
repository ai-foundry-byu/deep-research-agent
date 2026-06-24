"""Vendored, dependency-free copies of small utilities the agent relies on.

These were extracted from a larger agent framework so this package can stand
alone with no private-package imports:

- checkpoint.py / checkpoint_models.py — atomic file-based checkpoint/resume
- ollama_client.py — minimal local-Ollama router (used only by the `multi` profile)
"""
