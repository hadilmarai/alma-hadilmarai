import os
from groq import Groq

_client = None

def get_client():
    global _client

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. "
            "Make sure it is set in your environment or .env file."
        )

    if _client is None:
        _client = Groq(api_key=api_key)

    return _client

def call_llm(messages, temperature=0.4, model=None):
    client = get_client()
    model = model or os.getenv("MODEL", "openai/gpt-oss-120b")

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    return resp.choices[0].message.content
