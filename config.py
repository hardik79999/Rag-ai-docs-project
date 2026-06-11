from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_api_key: str
    vector_store: str = "chroma"          # "chroma" or "pgvector"
    chroma_persist_dir: str = "./chroma_db"

    # Only needed when vector_store = "pgvector"
    database_url: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
