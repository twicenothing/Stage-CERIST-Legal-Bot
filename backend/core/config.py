from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    CHROMA_PATH : str
    COLLECTION_NAME : str
    EMBEDDING_MODEL : str
    LLM_MODEL: str
    OLLAMA_HOST: str
    RERANKER_MODEL: str
    SQLALCHEMY_DATABASE_URL : str
    SECRET_KEY : str
    ALGORITHM : str
    ACCESS_TOKEN_EXPIRE_MINUTES : int
    PDF_PATH : str
    PDF_OLD_PATH : str
    class Config:
        env_file = "../.env"

settings = Settings() 

