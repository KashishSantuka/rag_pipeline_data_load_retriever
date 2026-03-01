# src/data_loader.py
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DataLoader:
    def __init__(self, filepath=None):
        """
        Initialize the data loader with the folder containing PDFs
        """
        if filepath is None:
            # Always resolves relative to this file, not the working directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(base_dir, "data", "text_files", "pdf")
        self.filepath = filepath

    def load_documents(self):
        """
        Load PDFs from the directory
        """
        try:
            print(f"Self filepath i want to see: {self.filepath}")
            loader = PyPDFDirectoryLoader(self.filepath)
            docs = loader.load()
            print(f"Loaded {len(docs)} documents from {self.filepath}")
            return docs
        except Exception as e:
            print(f"Error loading documents: {e}")
            raise

    def split_documents(self, documents, chunk_size=500, chunk_overlap=50):
        """
        Split documents into smaller chunks
        """
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_documents(documents)
            print(f"Split documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            print(f"Error splitting documents: {e}")
            raise