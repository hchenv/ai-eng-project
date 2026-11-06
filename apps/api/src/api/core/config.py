from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    OPENAI_API_KEY: str
    GROQ_API_KEY: str
    QDRANT_URL: str
    QDRANT_API_KEY: str
    LANGGRAPH_DB_URL: str
    TOOLS_DB_URL: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

config = Config()