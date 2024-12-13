import os
from typing import List, Optional
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import argparse
from tqdm import tqdm

class PDFVectorDBLoader:
    def __init__(
        self, 
        openai_api_key: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-ada-002"
    ):
        """Initialize the PDF Vector DB Loader
        
        Args:
            openai_api_key: OpenAI API key for embeddings
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks
            embedding_model: OpenAI embedding model to use
        """
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model=embedding_model
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        
    def load_pdfs(self, pdf_dir: str) -> List[str]:
        """Load PDFs from directory"""
        print(f"Loading PDFs from {pdf_dir}")
        loader = DirectoryLoader(
            pdf_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} PDF documents")
        return documents
    
    def process_documents(self, documents: List[str]) -> List[str]:
        """Split documents into chunks"""
        print("Splitting documents into chunks")
        texts = self.text_splitter.split_documents(documents)
        print(f"Created {len(texts)} text chunks")
        return texts
    
    def create_vector_db(
        self,
        texts: List[str],
        persist_directory: str,
        collection_name: Optional[str] = None
    ) -> Chroma:
        """Create and persist vector DB"""
        print("Creating vector database")
        
        # Create vector store
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        # Persist the database
        vectordb.persist()
        print(f"Vector database persisted to {persist_directory}")
        
        return vectordb

def main():
    parser = argparse.ArgumentParser(description='Load PDFs into a vector database')
    parser.add_argument('--pdf_dir', type=str, required=True, help='Directory containing PDF files')
    parser.add_argument('--persist_dir', type=str, required=True, help='Directory to persist vector DB')
    parser.add_argument('--openai_api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--collection_name', type=str, default=None, help='Name for the vector DB collection')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Size of text chunks')
    parser.add_argument('--chunk_overlap', type=int, default=200, help='Overlap between chunks')
    
    args = parser.parse_args()
    
    # Create persist directory if it doesn't exist
    os.makedirs(args.persist_dir, exist_ok=True)
    
    # Initialize loader
    loader = PDFVectorDBLoader(
        openai_api_key=args.openai_api_key,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    try:
        # Load and process PDFs
        documents = loader.load_pdfs(args.pdf_dir)
        texts = loader.process_documents(documents)
        
        # Create and persist vector DB
        vectordb = loader.create_vector_db(
            texts=texts,
            persist_directory=args.persist_dir,
            collection_name=args.collection_name
        )
        
        print("Successfully created vector database!")
        
    except Exception as e:
        print(f"Error creating vector database: {str(e)}")
        raise

if __name__ == "__main__":
    main() 