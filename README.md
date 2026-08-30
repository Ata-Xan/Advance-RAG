# Advance-RAG

## LLM gateway configuration

LLM calls are routed through Portkey, using the Groq provider slug `@rag1`.
Add the following to `.env` (the Groq API key remains stored only in Portkey):

```dotenv
PORTKEY_API_KEY=your_portkey_api_key
PORTKEY_GROQ_PROVIDER=rag1
GROQ_MODEL=your_groq_model_name
```

`PORTKEY_GROQ_PROVIDER` defaults to `rag1`, so it can be omitted when that is
the provider slug in your Portkey Model Catalog.
