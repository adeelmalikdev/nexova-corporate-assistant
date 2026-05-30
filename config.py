from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    GROK_API_KEY: str
    GROK_BASE_URL: str = "https://api.x.ai/v1"
    GROK_MODEL: str = "grok-5"
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_MINUTES: int = 480
    CHROMA_PERSIST_PATH: str = "./chroma_store"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANKED: int = 3
    BM25_WEIGHT: float = 0.4
    DENSE_WEIGHT: float = 0.6
    SIMILARITY_THRESHOLD: float = 0.3
    DEV_MODE: bool = False
    DATABASE_URL: str = "sqlite+aiosqlite:///./nexova.db"
    MAX_QUERY_LENGTH: int = 500
    MAX_LOGIN_ATTEMPTS: int = 5
    LOCKOUT_MINUTES: int = 15
    TEMP_PASSWORD_EXPIRY_HOURS: int = 48

settings = Settings()