"""
Shared pytest fixtures for ingestion tests.
Loads the embedding model ONCE for the entire test session
to avoid Windows access violation from repeated model loading.
"""

import pytest
from src.ingestion.embedder import DocumentEmbedder


@pytest.fixture(scope="session")
def shared_embedder():
    """
    Single DocumentEmbedder instance shared across ALL tests.
    scope="session" means it is created once and reused for
    every test in every test file in this session.
    """
    embedder = DocumentEmbedder()
    embedder._load_model()   # Force load immediately, not lazily
    return embedder
