import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

    # Groq credentials are stored in Portkey. The application only needs a
    # Portkey API key and the provider slug configured in the Portkey catalog.
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    PORTKEY_GROQ_PROVIDER = os.getenv("PORTKEY_GROQ_PROVIDER", "rag1")
    PORTKEY_GATEWAY_URL = os.getenv("PORTKEY_GATEWAY_URL", "https://api.portkey.ai/v1")

    GROQ_MODEL = os.getenv("GROQ_MODEL")


settings = Settings()
