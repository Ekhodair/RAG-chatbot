import os
from langchain.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    TextLoader, 
    UnstructuredHTMLLoader, 
    CSVLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
from helpers.constants import GLOBAL_CONFIG, CHROMA_DB_NAME


db_path = os.path.join(os.getcwd(), CHROMA_DB_NAME)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)

    
embedding_function = HuggingFaceEmbeddings(
    model_name=GLOBAL_CONFIG['retriever']['model_id'],
    model_kwargs={'device': 'cuda'})


vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding_function)


def load_and_split_document(file_path: str) -> List[Document]:
    """
    Load and split documents from various file types.
    Args:
        file_path (str): Path to the document file
    
    Returns:
        List[Document]: List of document chunks
    """
    file_ext = file_path.lower().split('.')[-1]
    loaders = {
        'pdf': PyPDFLoader,
        'docx': Docx2txtLoader,
        'txt': TextLoader,
        'html': UnstructuredHTMLLoader,
        'csv': CSVLoader,
        'md': UnstructuredMarkdownLoader
    }
    
    if file_ext not in loaders:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    try:
        loader = loaders[file_ext](file_path)
        documents = loader.load()
        return text_splitter.split_documents(documents)
    except Exception as e:
        raise RuntimeError(f"Error loading {file_path}: {str(e)}")


def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    """
    Index a document to Chroma vector store with a specific file ID.
    
    Args:
        file_path (str): Path to the document to be indexed
        file_id (int): Unique identifier for the document
    
    Returns:
        bool: True if indexing succeeds, False otherwise
    """
    try:
        splits = load_and_split_document(file_path)
        # Add metadata to each split
        for split in splits:
            split.metadata['file_id'] = file_id
        
        vectorstore.add_documents(splits)
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False


def delete_doc_from_chroma(file_id: int):
    """
    Delete all document chunks associated with a specific file ID from Chroma.
    
    Args:
        file_id (int): Unique identifier of the document to delete
    
    Returns:
        bool: True if deletion succeeds, False otherwise
    """
    try:
        docs = vectorstore.get(where={"file_id": file_id})
        print(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")
        
        vectorstore._collection.delete(where={"file_id": file_id})
        print(f"Deleted all documents with file_id {file_id}")
        
        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
        return False