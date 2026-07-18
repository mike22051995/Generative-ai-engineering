from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_file=".env")
    OPENAI_API_KEY:str
    DB_HOST:str
    DB_PORT:int
    DATABASE:str
    USER:str
    PASSWORD:str


settings=Settings()