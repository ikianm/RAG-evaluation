from src.core.chroma_db import ChromaDB
from src.core.document_handler import DocumentHandler
from src.core.rag_system import RAGSystem
from src.core.memory_handler import MemoryHandler
from dotenv import load_dotenv

load_dotenv(override=True)
        
def main():
    
    memory_handler = MemoryHandler()
    
    rag_system = RAGSystem(ChromaDB(), memory_handler)
    
    print("""
            \nQ&A Bot Initialized!
            (type /exit to end the conversation)
            """)
    while True:
        prompt = input('\nAsk a question: ')
        if prompt.strip().lower() == '/exit':
            break
        
        response = rag_system.generate_response(prompt)
        
        memory_handler.append_messages([
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': response}
        ])
        
        print(f'\nAssistant Response: {response}')
        
        if len(memory_handler.messages) > 6:
            memory_handler.summarize_messages()
        
        
        
if __name__ == '__main__':
    main()
    