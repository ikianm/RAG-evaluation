from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


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