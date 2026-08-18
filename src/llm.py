from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()



def create_llm():
    return ChatGroq(
        model = 'openai/gpt-oss-20b',
        temperature = 0
    )