from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from typing import List, Optional

class VectorDBRetriever:
    def __init__(
        self, 
        persist_directory: str,
        openai_api_key: str,
        collection_name: Optional[str] = None,
        embedding_model: str = "text-embedding-ada-002"
    ):
        """Initialize Vector DB retriever with existing DB
        
        Args:
            persist_directory: Directory where vector DB is stored
            openai_api_key: OpenAI API key for embeddings
            collection_name: Name of the collection to load
            embedding_model: OpenAI embedding model to use
        """
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model=embedding_model
        )
        
        # Load existing vector DB
        self.vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
    
    def retrieve(
        self, 
        query: str, 
        k: int = 3,
        filter: Optional[dict] = None
    ) -> List[str]:
        """Retrieve relevant documents for a query
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            filter: Optional filter criteria
            
        Returns:
            List of retrieved document contents
        """
        results = self.vectordb.similarity_search(
            query,
            k=k,
            filter=filter
        )
        return [doc.page_content for doc in results] 