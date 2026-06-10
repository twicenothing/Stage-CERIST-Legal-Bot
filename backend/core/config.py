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
    RAG_NUM_CTX: int
    RAG_NUM_PREDICT: int
    RAG_TEMPERATURE: float
    RAG_THINK: bool
    RAG_TOP_K_RETRIEVE: int
    RAG_TOP_K_RERANK: int
    VISION_TABLE_MODEL: str = "mistral-small3.1:latest"
    USE_PDF_VISION_FOR_TABLES: bool = True
    VISION_MAX_PAGES: int = 3
    VISION_PAGE_ZOOM: float = 3.0
    VISION_NUM_CTX: int = 32768
    VISION_NUM_PREDICT: int = 800


    class Config:
        env_file = "../.env"

settings = Settings() 

