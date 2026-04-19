from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_provider: str = "anthropic"
    llm_model: str = "claude-3-5-haiku-20241022"

    embedding_provider: str = "local"
    embedding_model: str = "all-MiniLM-L6-v2"

    chunk_size: int = 800
    chunk_overlap: int = 150
    retrieval_k: int = 4

    vectorstore_path: str = "./vectorstore"


settings = Settings()
