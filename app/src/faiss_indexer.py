import numpy as np
import faiss

class FaissIndexer:
    """A class to create and search a FAISS index for efficient similarity search."""
    def __init__(self, embedding_size: int):
        self.index = faiss.IndexFlatL2(embedding_size)

    def add_items(self, item_embeddings: np.ndarray) -> None:
        """Adds item embeddings to the FAISS index."""
        self.index.add(item_embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> np.ndarray:
        """Searches the FAISS index for the top-k most similar items to the query embedding."""
        _, indices = self.index.search(query_embedding, top_k)
        return indices[0]