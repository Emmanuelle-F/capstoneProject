from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List
import numpy as np


class EmbeddingGenerator:

    # Converte resume/job description text into vector embeddings

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):

        try:
            self.embedding_model = HuggingFaceEmbedding(model_name=model_name)
            print(f"[INFO] Successfully loaded embedding model: {model_name}")
        except Exception as error:
            print(f"[ERROR] Failed to load embedding model: {error}")
            raise error


    def embed_text(self, text: str) -> np.ndarray:

        if not text or len(text.strip()) == 0:
            raise ValueError("[ERROR] Empty text received for embedding.")

        try:
            embedding = self.embedding_model.get_text_embedding(text)
            return np.array(embedding)
        except Exception as error:
            print(f"[ERROR] Failed to generate embedding: {error}")
            return np.array([])


    # def embed_multiple(self, text_list: List[str]) -> List[np.ndarray]:
    #     """
    #     Generate embeddings for multiple text documents.

    #     Args:
    #         text_list (List[str]): List of resume or job description texts.

    #     Returns:
    #         List[np.ndarray]: List of embedding vectors.
    #     """
    #     if not text_list:
    #         raise ValueError("[ERROR] Empty list received for batch embedding.")

    #     embeddings = []

    #     for index, text in enumerate(text_list):
    #         try:
    #             vector = self.embed_text(text)
    #             embeddings.append(vector)
    #         except Exception as error:
    #             print(f"[ERROR] Skipping text at index {index}: {error}")
    #             embeddings.append(np.array([]))

    #     return embeddings
