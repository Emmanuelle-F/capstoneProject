import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import numpy as np

# Load your API key from .env
load_dotenv()


class PineconeManager:
    """
    Manages all Pinecone operations for the AI Resume Assistant project:
    - Connect to Pinecone
    - Create a serverless index (if needed)
    - Upsert resume embeddings
    - Query similar resumes based on embeddings
    """

    def __init__(self, index_name: str, dimension: int = 384):
        """
        Initialize Pinecone client and connect/create index.

        Args:
            index_name (str): Name of the Pinecone index.
            dimension (int): Size of embedding vectors.
        """
        api_key = os.getenv("PINECONE_API_KEY")

        if not api_key:
            raise ValueError(
                "[ERROR] PINECONE_API_KEY not found in .env file. "
                "Please set it before running the application."
            )

        # Initialize Pinecone v3 client
        self.client = Pinecone(api_key=api_key)

        # Check for existing indexes
        existing_indexes = self.client.list_indexes().names()

        # Create the index only if it doesn't exist
        if index_name not in existing_indexes:
            print(f"[INFO] Creating serverless Pinecone index '{index_name}'...")

            try:
                self.client.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",        # could be "gcp"
                        region="us-east-1"  # stable region for serverless
                    )
                )
            except Exception as error:
                raise RuntimeError(f"[ERROR] Index creation failed: {error}")

        # Connect to the index
        try:
            self.index = self.client.Index(index_name)
            print(f"[INFO] Connected to Pinecone index '{index_name}'.")
        except Exception as error:
            raise RuntimeError(f"[ERROR] Failed to connect to index: {error}")


    def upsert_resume(self, embedding: np.ndarray, resume_id: str, metadata: dict):
        """
        Insert or update a resume embedding in Pinecone.

        Args:
            embedding (np.ndarray): Resume embedding vector.
            resume_id (str): Unique ID (usually filename).
            metadata (dict): Additional info (e.g., raw text).
        """
        if embedding is None or embedding.size == 0:
            print(f"[WARNING] Skipping empty embedding for: {resume_id}")
            return

        try:
            self.index.upsert([
                {
                    "id": resume_id,
                    "values": embedding.tolist(),
                    "metadata": metadata
                }
            ])

            print(f"[INFO] Successfully upserted resume: {resume_id}")

        except Exception as error:
            print(f"[ERROR] Pinecone upsert failed for {resume_id}: {error}")


    def query_similar(self, query_vector: np.ndarray, top_k: int = 5):
        """
        Query Pinecone index to find the most relevant resumes.

        Args:
            query_vector (np.ndarray): Embedding of job description.
            top_k (int): Number of top results to return.

        Returns:
            dict: Pinecone query response with matches.
        """
        if query_vector is None or query_vector.size == 0:
            print("[ERROR] Empty query vector. Cannot perform search.")
            return None

        try:
            response = self.index.query(
                vector=query_vector.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            return response

        except Exception as error:
            print(f"[ERROR] Pinecone query failed: {error}")
            return None
