from typing import List
import chromadb
from chromadb.config import Settings
from .exceptions import RetrieverError
from .utils import ProcessingConfig

class Retriever:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.client = chromadb.Client(Settings(
            allow_reset=True,
            is_persistent=True,
            persist_directory="chroma_db"
        ))
        self._initialize_collection()

    def _initialize_collection(self):
        """Initialize or get the collection"""
        try:
            # Try to delete existing collection if it exists
            try:
                self.client.delete_collection("document_chunks")
            except:
                pass
            # Create new collection
            self.collection = self.client.create_collection("document_chunks")
        except Exception as e:
            raise RetrieverError(f"Error initializing collection: {e}")

    def add_chunks(self, chunks: List[str], embeddings: List[List[float]], metadata: List[dict] = None):
        """Add text chunks to ChromaDB"""
        try:
            # Create default metadata if none provided
            if metadata is None:
                metadata = [{"chunk_id": str(i), "position": i} for i in range(len(chunks))]
            
            # Generate unique IDs for each chunk
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadata,
                ids=ids
            )
        except Exception as e:
            raise RetrieverError(f"Error adding chunks to ChromaDB: {e}")

    def retrieve_relevant(self, query: str, embeddings: List[float]) -> List[str]:
        """Retrieve relevant chunks based on query"""
        try:
            results = self.collection.query(
                query_embeddings=[embeddings],
                n_results=self.config.retrieval_k
            )
            return results['documents'][0]  # First list of documents
        except Exception as e:
            raise RetrieverError(f"Error retrieving chunks from ChromaDB: {e}")

    def reset(self):
        """Reset the collection by recreating it"""
        try:
            self._initialize_collection()
        except Exception as e:
            raise RetrieverError(f"Error resetting ChromaDB collection: {e}")