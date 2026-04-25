import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "llama-3.1-8b-instant"

    CHUNK_SIZE = 100
    CHUNK_OVERLAP = 20

    VECTOR_DB_PATH = "./chroma_db"

    DATA_PATH = "./data/documents"
    GOLDEN_DATASET = "./data/golden_dataset.json"