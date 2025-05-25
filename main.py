from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

class DocumentHandler:
    
    def __init__(self):
        self.loader = DirectoryLoader('./files', glob='**/*.txt', loader_cls=TextLoader)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, 
            chunk_overlap=40,
            separators=["\n\n", "\n", "۔", "؟", "!", " ", ""], 
            keep_separator=True
            )
        self.documents = []
            
    def load_documents(self) -> None:
        try:
            self.documents = self.loader.load()
        except Exception as e:
            print(f'Error loading file {str(e)}')
            raise e
            
    def split_documents(self) -> list[Document]:
        try:
            splitted_documents = self.text_splitter.split_documents(self.documents)
            return splitted_documents
        except Exception as e:
            print(f'Error splitting the documents {str(e)}')
            raise e
        
        
class ChromaDB:
    
    def __init__(self):
        try:
            self.retriever = Chroma(
                    collection_name='system-data',
                    embedding_function=OllamaEmbeddings(model='nomic-embed-text'),
                    persist_directory='./chroma_data'
                ).as_retriever(search_kwargs={'k': 5})
        except Exception as e:
            print(f'Failed to setup ChromaDB {str(e)}')
            
    def add_documents(self, splitted_documents: list[Document]):
        try:
            self.retriever.add_documents(splitted_documents)
        except Exception as e:
            print(f'\nError adding documents to vector store {str(e)}')
            raise e
            
    def get_related_documents(self, query: str) -> list[Document]:
        try:
            relevant_documents = self.retriever.invoke(query)
            return relevant_documents
        except Exception as e:
            print(f'Error retrieving related documents {str(e)}')
            raise e
        
class LLM:
    
    def __init__(self, temperature: float):
        self.chat_model = ChatOpenAI(
            api_key=os.getenv('OPENROUTER_API_KEY'),
            base_url=os.getenv('OPENROUTER_BASE_URL'),
            model='openai/gpt-3.5-turbo',
            temperature=temperature
        )
        
class MemoryHandler:
    def __init__(self):
        self.messages = []
        self.llm = LLM(temperature=0.0).chat_model
        
    def append_message(self, message: str): 
        self.messages.append(message)
        
    def summarize_messages(self):
        summarization_prompt = PromptTemplate.from_template("""
            **وظیفه خلاصه‌سازی گفتگو**  
            شما یک دستیار هوشمند هستید که وظیفه خلاصه‌کردن گفتگوی بین کاربر و دستیار هوشمند را بر عهده دارید.  
            خلاصه‌ای مختصر اما آموزنده ایجاد کنید که موارد زیر را پوشش دهد:  

            1. **موضوعات کلیدی** مطرح‌شده  
            2. **سوالات مهم** کاربر  
            3. **اطلاعات یا پاسخ‌های کلیدی** ارائه‌شده توسط دستیار  
            4. **تصمیمات یا نتیجه‌گیری‌های** گرفته‌شده (در صورت وجود)  
            5. **سوالات بی‌پاسخ یا موضوعات حل‌نشده** (اگر وجود دارد)  

            **راهنما:**  
            - از دیدگاه **بیطرف و سوم‌شخص** بنویسید.  
            - جزئیات ضروری برای **حفظ زمینه آینده** را حفظ کنید.  
            - از **زبان ساده و واضح** استفاده کنید.  
            - **گفتگوهای غیرمرتبط** (مثل احوالپرسی) را نادیده بگیرید، مگر اینکه مهم باشند.  

            **تاریخچه گفتگو:**  
            {conversation_history}  

            **خلاصه:**  
            """)
                    
        concatinated_messages = ' '.join(self.messages[-10:])
        
        summarization_chain = summarization_prompt | self.llm | StrOutputParser()
        summary = summarization_chain.invoke({'conversation_history': concatinated_messages})
        
        self.messages = self.messages[:-5]
        self.messages.append(summary)

class RAGSystem:
    
    def __init__(self):
        self.llm = LLM(temperature=0.3).chat_model
        
    def create_chat_prompt(
        self, 
        input_prompt: str, 
        related_documents: list[Document],
        memory: list[str]
        ) -> ChatPromptTemplate:
        
        related_texts = [doc.page_content for doc in related_documents]
        context = ''.join(related_texts)
        conversation_history = ''.join(memory)
        chat_prompt_template = ChatPromptTemplate.from_messages([
            ('system', 'شما یک دستیار هوشمند و مفید هستید.'),
            ('user', """
                شما یک دستیار هوشمند هستید. با استفاده از اطلاعات ارائه‌شده در «متن مرتبط» و «تاریخچه گفتگو»، به سوال کاربر پاسخ دهید. اگر پاسخ در این منابع یافت نشد، از دانش خود برای ارائه پاسخ دقیق و مختصر استفاده کنید. لحن پاسخ باید حرفه‌ای و بیطرف باشد. از توضیحات غیرضروری یا زبان غیررسمی خودداری کنید. به خودتان یا منبع دانشتان اشاره نکنید.

                تاریخچه گفتگو:
                {conversation_history}

                متن مرتبط:
                {context}

                سوال کاربر:
                {question}

                **پاسخ:**
            """)
        ])
        
        return chat_prompt_template.invoke({
            'conversation_history': conversation_history,
            'context': context,
            'question': input_prompt
            })
    
    def generate_response(self, chat_prompt_template: ChatPromptTemplate):
        if not self.llm:
            raise ValueError('No initialized LLM found')
        try:
            response = self.llm.invoke(chat_prompt_template)
            return response.content
        except Exception as e:
            print(f'Error generating a response {str(e)}')
            raise e    
        
        
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
    