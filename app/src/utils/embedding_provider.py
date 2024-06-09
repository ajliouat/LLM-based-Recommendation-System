import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingProvider:
    """A class to generate embeddings for text queries using a pre-trained sentence transformer model."""
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, query: str) -> np.ndarray:
        """Generates the embedding for a given text query."""
        return self.model.encode(query)