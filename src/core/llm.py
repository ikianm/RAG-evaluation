from langchain_openai import ChatOpenAI

import os

class LLM:
    
    def __init__(self, temperature: float):
        self.chat_model = ChatOpenAI(
            model='google/gemini-2.5-flash-preview-05-20',
            api_key=os.getenv('OPENROUTER_API_KEY'),
            base_url=os.getenv('OPENROUTER_BASE_URL'),
            temperature=temperature
        )