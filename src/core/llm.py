from langchain_openai import ChatOpenAI

import os

class LLM:
    
    def __init__(self, temperature: float):
        self.chat_model = ChatOpenAI(
            api_key='ollama',
            base_url=os.getenv('OLLAMA_BASE_URL'),
            model='llama3.2',
            temperature=temperature
        )