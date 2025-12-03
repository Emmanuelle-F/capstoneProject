import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import numpy as np

load_dotenv()

class PineconeManager:

    def __init__(self, index_name: str, dimension: int = 384):

        api_key = os.getenv("PINECONE_API_KEY")

        if not api_key:
            raise ValueError(
                "[ERROR] PINECONE_API_KEY not found in .env file. "
                "Please set it before running the application."
            )

        # Initialize Pinecone
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
                        cloud="aws",       
                        region="us-east-1" 
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
        
        # Insert or update a resume embedding in Pinecone.

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

    def query_similar(self, query_vector: np.ndarray, top_k: int = 8):
        
        # Query Pinecone index to find the most relevant resumes.

        if query_vector is None or query_vector.size == 0:
            print("[ERROR] Empty query vector. Cannot perform search.")
            return None

        try:
            response = self.index.query(
                vector=query_vector.tolist(),
                top_k=top_k,
                include_metadata=True,
            )
            return response

        except Exception as error:
            print(f"[ERROR] Pinecone query failed: {error}")
            return None


    def clear_index(self) -> None:
 
        try:
            self.index.delete(delete_all=True)
            print("[INFO] Pinecone index cleared (delete_all=True).")

        except TypeError:
            self.index.delete(deleteAll=True)
            print("[INFO] Pinecone index cleared (deleteAll=True fallback).")

        except Exception as error:
            print(f"[ERROR] Failed to clear Pinecone index: {error}")
            raise
