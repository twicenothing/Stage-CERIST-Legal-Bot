from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    CHROMA_PATH : str
    COLLECTION_NAME : str
    EMBEDDING_MODEL : str
    LLM_MODEL: str
    OLLAMA_HOST: str
    class Config:
        env_file = "../.env"

settings = Settings() 

