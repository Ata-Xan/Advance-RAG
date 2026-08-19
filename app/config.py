import os
from dotenv import load_dotenv


load_dotenv()

class Settings:
    GEMINI_API_KEY= os.getenv("GEMINI_API_KEY")
    QDRANT_URL=os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION=os.getenv("QDRANT_COLLECTION")
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL")
    GROQ_FALLBACK_API_KEY=os.getenv("GROQ_FALLBACK_API_KEY")
    

settings = Settings()
    
    
    


    
    