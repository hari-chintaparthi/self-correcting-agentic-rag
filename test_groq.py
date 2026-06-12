from dotenv import load_dotenv
import os
import sys

# Force load the Desktop env file
env_path = r"C:\Users\C Hari\OneDrive\Desktop\self-correcting-rag\.env"
load_dotenv(env_path)

key = os.getenv("GROQ_API_KEY")
print(f"Loaded Key: {key}")
if key:
    print(f"Key Length: {len(key)}")
    print(f"Starts with gsk: {key.startswith('gsk_')}")

from langchain_groq import ChatGroq

try:
    print("Initializing ChatGroq...")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    print("Sending test request 'Hello'...")
    res = llm.invoke("Hello")
    print("--- SUCCESS ---")
    print("Response:", res.content)
except Exception as e:
    print("--- FAILURE ---")
    print("Error:", str(e))
