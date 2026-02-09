from pathlib import Path
import os
from dotenv import load_dotenv   

load_dotenv()

# Paths 
BASE_DIR = Path(__file__).resolve().parent.parent
DOC_PATH = BASE_DIR / "data" / "Pathfinder AI - Training Materials.docx"

# Supported Inputs
SUPPORTED_AGE_RANGES = [
    "18-24", "25-34", "35-44", "45-54", "55-64", "65+"
]

SUPPORTED_INCOME_TYPES = [
    "Salary",
    "Salary + Bonus",
    "Hourly + Overtime",
    "Commission",
    "Self-Employed",
    "Other Employment Income"
]

# Retrieval 
TOP_K = 5
BM25_TOP_N = 10


# LLM Provider 
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b").strip()

