from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # ── LLM Provider ───────────────────────────
    LLM_PROVIDER: str = "google"

    # ── Google Gemini ──────────────────────────
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # ── OpenAI ─────────────────────────────────
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"

    # ── Anthropic ──────────────────────────────
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-sonnet-4-5"

    # ── Azure OpenAI ───────────────────────────
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_DEPLOYMENT: Optional[str] = None
    AZURE_OPENAI_API_VERSION: str = "2024-02-01"

    # ── LLM Settings ───────────────────────────
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048

    # ── Embedding Settings ─────────────────────
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: str = "cpu"

    # ── Reranker Settings ──────────────────────
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_TOP_K: int = 40

    # ── Chunking Settings ──────────────────────
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 150
    CHUNK_MIN_SIZE: int = 50

    # ── Retrieval Settings ─────────────────────
    MAX_CHUNKS_DENSE: int = 20
    MAX_CHUNKS_BM25: int = 20
    TOP_K_FINAL: int = 5
    SIMILARITY_THRESHOLD: float = 0.0

    # ── Storage Paths ──────────────────────────
    CHROMA_DB_PATH: str = "./data/chroma_db"
    BM25_INDEX_PATH: str = "./data/bm25_index.pkl"
    EVALUATION_DB_PATH: str = "./data/evaluation.db"

    # ── Evaluation Thresholds ──────────────────
    GROUNDEDNESS_THRESHOLD: float = 0.75
    CITATION_THRESHOLD: float = 0.85
    RELEVANCE_THRESHOLD: float = 0.70
    MAX_REGENERATIONS: int = 2

    # ── Logging ────────────────────────────────
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def get_embedding_model_name(self) -> str:
        return self.EMBEDDING_MODEL

    def get_llm_provider(self) -> str:
        return self.LLM_PROVIDER.lower()


settings = Settings()
