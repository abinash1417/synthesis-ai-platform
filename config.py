import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Central configuration for the Synthesis application."""

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MODEL_NAME = "openai/gpt-oss-120b"
    MODEL_TEMPERATURE = 0.3

    CHROMA_DB_PATH = "./synthesis_db"
    COLLECTION_NAME = "documents"

    MAX_CHUNK_SIZE = 500
    MIN_CHUNK_LENGTH = 30
    SEARCH_RESULTS_COUNT = 4

    MAX_REVIEW_ATTEMPTS = 2
    RETRY_MIN_WAIT = 4
    RETRY_MAX_WAIT = 30
    RETRY_MAX_ATTEMPTS = 5

    APP_TITLE = "🔬 Synthesis"
    APP_CAPTION = "AI Research & Knowledge Platform — upload documents, ask questions, get agent-researched answers"
    PAGE_TITLE = "Synthesis — AI Research Platform"