from .src.core.document_handler import DocumentHandler
from .src.core.chroma_db import ChromaDB

document_handler = DocumentHandler()
    
chroma_db = ChromaDB()

document_handler.load_documents()

splitted_documents = document_handler.split_documents()

chroma_db.add_documents(splitted_documents)

