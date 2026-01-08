from __future__ import annotations

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings
from pydantic import model_validator
import os


class Settings(BaseSettings):
    """Project-wide configuration loaded from environment variables (.env optional)."""

    # General settings
    PORT: int = Field(8000, description="HTTP port for MCP server")
    LOG_LEVEL: str = Field("DEBUG", description="Root log level for Loguru")
    MCP_SERVER_URL: HttpUrl = Field("http://localhost:8000", description="URL of the Core MCP server (Gateway)")

    # LLM / Ollama
    OLLAMA_BASE_URL: HttpUrl = Field("http://localhost:11434", description="Base URL of the local Ollama server")
    LLM_MODEL: str = Field("qwen3:8b", description="Model name passed to Ollama")
    SLM_MODEL: str = Field("qwen3:0.6b", description="Smaller, faster model for simple tasks via call_slm tool")
    EMBEDDING_MODEL: str = Field("embeddinggemma:latest", description="Ollama model for text embeddings")
    EMBEDDING_SIZE: int | None = Field(None, description="Ollama embedding size")
    TEMPERATURE: float = Field(0.5, description="LLM sampling temperature")
    TOP_P: float = Field(0.9, description="Nucleus sampling parameter (probability mass)")
    TOP_K: int = Field(40, description="Top-K sampling parameter (number of candidates)")
    MAX_TOOLS_ITERATIONS: int = Field(15, description="Max reasoning/tool iterations per user prompt")
    CONTEXT_WINDOW: int | None = Field(None, description="Manual cap for the LLM context window (tokens). If None, uses model default.")

    # UI (CLI) options
    UI_THEME: str = Field("dark", description="CLI theme (dark/light)")
    SHOW_THOUGHTS: bool = Field(True, description="Whether to stream the model's <think> content")

    # Database
    SESSION_DB_PATH: str = Field("database/sessions.db", description="Path to SQLite database that stores message history")

    # Vector Memory
    VECTOR_DB_PATH: str = Field("database/vector_memory", description="Path to directory for vector memory files (FAISS index and metadata)")
    VECTOR_SEARCH_TOP_K: int = Field(5, description="Number of results to return from vector search")
    VECTOR_SEARCH_RERANK: bool = Field(False, description="Enable reranking of vector search results")

    # Security – MUST be provided via environment or .env
    # Preferred variable name is **MCP_JWT_KEY**.  For backward compatibility the
    # old env var ``MCP_HMAC_KEY`` is still recognised but should be considered
    # deprecated.

    MCP_JWT_KEY: str | None = Field(
        default=None,
        alias="MCP_JWT_KEY",  # primary env name
        description="Shared secret used to sign/verify tool results via JWT (HS256)",
    )

    # Allow legacy env var
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
        "env_prefix": "",
        "env_allow_population_by_field_name": True,
        "populate_by_name": True,
        "env": {
            "MCP_HMAC_KEY": "MCP_JWT_KEY",
        },
    }

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def _ensure_secret_set(self):  # noqa: D401 – pydantic hook
        """Fail fast if JWT signing secret is missing."""
        if not self.MCP_JWT_KEY:
            # Legacy fallback
            legacy = os.getenv("MCP_HMAC_KEY")
            if legacy:
                object.__setattr__(self, "MCP_JWT_KEY", legacy)
            else:
                raise ValueError(
                    "MCP_JWT_KEY is not set. Define it via environment variable (.env or export) before running APE."
                )
        return self
    


    CONTEXT_MARGIN_TOKENS: int = Field(1024, description="Safety buffer deducted from context window before pruning/summarization")

    # Memory / summarisation
    SUMMARIZE_THOUGHTS: bool = Field(
        False,
        description="When True, <think> blocks are included in summarisation; when False they are stripped before summarize_text."
    )

    SUMMARY_MAX_TOKENS: int = Field(
        128,
        description="Default maximum length (in tokens) for summaries produced by summarize_text when caller does not specify a cap.",
        ge=32,
        le=8192,
    )


settings = Settings() 