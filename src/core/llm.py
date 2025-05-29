from langchain_openai import ChatOpenAI
import os

class LLM:
    
    def __init__(self, temperature: float):
        self.chat_model = ChatOpenAI(
            model=os.getenv('LLM_MODEL'),
            base_url=os.getenv('LLM_BASE_URL'),
            api_key=os.getenv('LLM_API_KEY'),
            temperature=temperature
        )