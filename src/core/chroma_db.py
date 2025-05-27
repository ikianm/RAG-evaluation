from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

import os

class ChromaDB:
    
    def __init__(self):
        self.embedding = HuggingFaceEmbeddings(
            model_name=os.getenv('EMBEDDING_MODEL'),
            model_kwargs={"trust_remote_code": True}
            )
        
        try:
            self.retriever = Chroma(
                    collection_name='system-data',
                    embedding_function=self.embedding,
                    persist_directory='./chroma_data'
                ).as_retriever(search_kwargs={'k': 5})
        except Exception as e:
            print(f'Failed to setup ChromaDB {str(e)}')
            
    def add_documents(self, splitted_documents: list[Document]):
        print('\nAdding documents to vector store...')
        try:
            self.retriever.add_documents(splitted_documents)
            print(f'\n{len(splitted_documents)} documents added successfully!')
        except Exception as e:
            print(f'\nError adding documents to vector store {str(e)}')
            raise e
            
    def get_relevant_documents(self, query: str) -> list[Document]:
        try:
            relevant_documents = self.retriever.invoke(query)
            return relevant_documents
        except Exception as e:
            print(f'Error retrieving related documents {str(e)}')
            raise e