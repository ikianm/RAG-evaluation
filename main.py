from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

class DocumentHandler:
    
    def __init__(self):
        self.loader = DirectoryLoader('./text_files', glob='**/*.txt', loader_cls=TextLoader)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50,
            separators=['\n\n', '\n', '.', ',', ' ', '']
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
        summarization_prompt = PromptTemplate.from_template( """**Conversation Summary Task**
                You are an AI assistant tasked with summarizing a conversation between a user and an AI assistant. 
                Create a concise yet informative summary that captures:

                1. The key topics discussed
                2. Important questions asked by the user
                3. Key information or answers provided by the AI
                4. Any decisions made or conclusions reached
                5. Open questions or unresolved topics (if any)

                **Guidelines:**
                - Maintain an objective, third-person perspective
                - Preserve crucial details needed for future context
                - Use clear, simple language
                - Ignore small talk or greetings unless relevant

                **Conversation History:**
                {conversation_history}

                **Summary:**""")
        
        concatinated_messages = ' '.join(self.messages[-10:])
        
        summarization_chain = summarization_prompt | self.llm | StrOutputParser()
        summary = summarization_chain.invoke({'conversation_history': concatinated_messages})
        
        self.messages = self.messages[:-5]
        self.messages.append(summary)
        print('SUMMARIZED LAST 5 MESSAGES: ', summary)

class RAGSystem:
    
    def __init__(self):
        self.llm = LLM(temperature=0.3).chat_model
        
    def create_chat_prompt(
        self, input_prompt: str, 
        related_documents: list[Document],
        memory: list[str]
        ) -> ChatPromptTemplate:
        related_texts = [doc.page_content for doc in related_documents]
        context = ''.join(related_texts)
        conversation_history = ''.join(memory)
        chat_prompt_template = ChatPromptTemplate.from_messages([
            ('system', 'You are a helpful assistant'),
            ('user', """
               You are an AI assistant. Answer the user's question using the information provided in the context and the previous conversation. If the answer cannot be found in either, use your own knowledge to provide a concise and accurate response. Maintain a professional and neutral tone. Avoid unnecessary elaboration or informal language. Do not refer to yourself or the source of your knowledge.

                Conversation history:
                {conversation_history}

                Context:
                {context}

                User question:
                {question}
                
                **Answer:**
                """)
        ])
        
        return chat_prompt_template.format_messages({'context': context, 'conversation_history': conversation_history,'question': input_prompt})
    
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
    
    document_handler.load_documents()
    splitted_documents = document_handler.split_documents()
    print('\nDocuments loaded!')
    
    chroma_db.add_documents(splitted_documents)
    print('\nDocuments loaded into Vector Store!')
    
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