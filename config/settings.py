from dotenv import load_dotenv
import os

load_dotenv()

# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"

# Agent - PRODUCTION SETTINGS
MAX_ITERATIONS = 7
CONFIDENCE_THRESHOLD = 0.65  # ← Achievable with real data

# MLflow — SQLite backend
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "wet-lab-agent"

# PubMed
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"