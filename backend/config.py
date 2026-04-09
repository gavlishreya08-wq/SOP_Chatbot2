from pathlib import Path
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # LLM
    llm_provider: str = "gemini"  # "gemini" or "groq"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-pro"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # Admin
    admin_password: str = "admin123"
    jwt_secret: str = "change-me-in-production"

    # Paths
    sop_documents_dir: str = str(PROJECT_ROOT / "sop_documents")
    flowcharts_dir: str = str(PROJECT_ROOT / "flowcharts")
    img_txt_dir: str = str(PROJECT_ROOT / "img_txt")
    chroma_db_dir: str = str(PROJECT_ROOT / "chroma_db")
    sop_metadata_path: str = str(PROJECT_ROOT / "sop_metadata.json")

    # Sync
    sop_base_url: str = "https://upaygoa.com/geltm/helpndoc"

    # Data (feedback, analytics, conversations)
    data_dir: str = str(PROJECT_ROOT / "data")

    model_config = {"env_file": str(PROJECT_ROOT / ".env"), "extra": "ignore"}


settings = Settings()
