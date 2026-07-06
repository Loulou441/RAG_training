import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SENTENCE_TRANSFORMERS_MODEL = "all-MiniLM-L6-v2" 
LLM_MODEL = "openai/gpt-oss-20b"  # Modèle LLM pour la génération de réponses