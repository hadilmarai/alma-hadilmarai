import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

resp = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "system", "content": "You are a collaborative tutor."},
        {"role": "user", "content": "Ask me two clarifying questions about a buggy Python function."}
    ],
    temperature=0.3,
)

print(resp.choices[0].message.content)
