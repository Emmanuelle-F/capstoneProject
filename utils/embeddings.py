from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List
import numpy as np


class EmbeddingGenerator:
    """
    A helper class to generate text embeddings using a HuggingFace model.
    This module is responsible for converting resume/job description text
    into vector embeddings that can be stored in a vector database (Pinecone).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):

        #  sentence-transformers/all-MiniLM-L6-v2: maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.
        # other models: BAAI/bge-large-en-v1.5

        """
        Initialize the embedding model.

        Args:
            model_name (str): Name of the HuggingFace embedding model.
        """
        try:
            self.embedding_model = HuggingFaceEmbedding(model_name=model_name)
            print(f"[INFO] Successfully loaded embedding model: {model_name}")
        except Exception as error:
            print(f"[ERROR] Failed to load embedding model: {error}")
            raise error


    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embeddings for a single text document.

        Args:
            text (str): The text to embed.

        Returns:
            np.ndarray: The generated embedding vector.
        """
        if not text or len(text.strip()) == 0:
            raise ValueError("[ERROR] Empty text received for embedding.")

        try:
            embedding = self.embedding_model.get_text_embedding(text)
            return np.array(embedding)
        except Exception as error:
            print(f"[ERROR] Failed to generate embedding: {error}")
            return np.array([])


    def embed_multiple(self, text_list: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple text documents.

        Args:
            text_list (List[str]): List of resume or job description texts.

        Returns:
            List[np.ndarray]: List of embedding vectors.
        """
        if not text_list:
            raise ValueError("[ERROR] Empty list received for batch embedding.")

        embeddings = []

        for index, text in enumerate(text_list):
            try:
                vector = self.embed_text(text)
                embeddings.append(vector)
            except Exception as error:
                print(f"[ERROR] Skipping text at index {index}: {error}")
                embeddings.append(np.array([]))

        return embeddings
