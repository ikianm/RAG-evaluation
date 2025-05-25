from src.core.chroma_db import ChromaDB
from src.core.document_handler import DocumentHandler
from src.core.rag_system import RAGSystem
from src.core.memory_handler import MemoryHandler
from dotenv import load_dotenv

load_dotenv()
        
def main():
    document_handler = DocumentHandler()
    print('Document handler initialized!')
    
    chroma_db = ChromaDB()
    print('\nChroma DB initialized!')
    
    # document_handler.load_documents()
    # print('\nDocuments loaded!')
    
    # splitted_documents = document_handler.split_documents()
    # print('\nDocuments splitted!')

    # chroma_db.add_documents(splitted_documents)
    # print('\nDocuments loaded into Vector Store!')
    
    memory_handler = MemoryHandler()
    print('\nMemory handler initialized!')
        
    rag_system = RAGSystem()
    print('\nRAG System Initialized!')
    print("""
            \nQ&A Bot Initialized!
            (type /exit to end the conversation)
            """)
    
    while True:
        input_prompt = input('\nAsk a question: ')
        if input_prompt.strip().lower() == '/exit':
            break
        
        related_documents = chroma_db.get_related_documents(input_prompt)
        chat_prompt_template = rag_system.create_chat_prompt(input_prompt=input_prompt, related_documents=related_documents, memory=memory_handler.messages)
        response = rag_system.generate_response(chat_prompt_template)
        
        memory_handler.append_message(f'Human Question: {input_prompt}')
        memory_handler.append_message(f'AI Response: {response}')
        
        if len(memory_handler.messages) > 10:
            memory_handler.summarize_messages()
        
        print(f'\nAssistant Response: {response}')
        
        
if __name__ == '__main__':
    main()
    