from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentHandler:
    
    def __init__(self):
        self.loader = DirectoryLoader('./files', glob='**/*.txt', loader_cls=TextLoader)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50,
            separators=["\n\n", "\n", "۔", "؟", "!", " ", ""], 
            keep_separator=True
            )
        self.documents = []
            
    def load_documents(self) -> None:
        print('\nLoading documents from files...')
        try:
            self.documents = self.loader.load()
            print(f'\nLoaded {len(self.documents)} documents successfully!')
        except Exception as e:
            print(f'Error loading file {str(e)}')
            raise e
            
    def split_documents(self) -> list[Document]:
        print('\nSplitting documents into chunks...')
        try:
            splitted_documents = self.text_splitter.split_documents(self.documents)
            print(f'\nSplitted documents to {len(splitted_documents)} chunks successfully!')
            return splitted_documents
        except Exception as e:
            print(f'Error splitting the documents {str(e)}')
            raise e