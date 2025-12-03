from typing import List, Dict, Any
from utils.embeddings import EmbeddingGenerator
from utils.pinecone_utils import PineconeManager


class ResumeRanker:
    
    # Handles semantic similarity scoring between resumes and the job description.
    def __init__(self, index_name: str = "resume-index", embedding_dim: int = 384) -> None:
        
        # Initialize embedder and Pinecone manager.
        self.embedder = EmbeddingGenerator()
        self.pinecone_manager = PineconeManager(
            index_name=index_name,
            dimension=embedding_dim
        )

    def rank_resumes(self, job_description: str, top_k: int = 10) -> List[Dict[str, Any]]:
        
        # Rank resumes based on semantic similarity.
        if not job_description or len(job_description.strip()) == 0:
            raise ValueError("[ERROR] Job description cannot be empty.")

        # Embed job description
        try:
            job_embedding = self.embedder.embed_text(job_description)
        except Exception as error:
            print(f"[ERROR] Failed to embed job description: {error}")
            return []

        # Query Pinecone
        pinecone_response = self.pinecone_manager.query_similar(
            query_vector=job_embedding,
            top_k=top_k,
        )

        if not pinecone_response:
            print("[WARNING] Empty response from Pinecone.")
            return []

        matches = getattr(pinecone_response, "matches", None)
        if not matches:
            print("[WARNING] No matching resumes found.")
            return []

        ranked_results: List[Dict[str, Any]] = []

        for match in matches:
            try:
               
                resume_id = getattr(match, "id", "unknown_id")
                raw_score = float(getattr(match, "score", 0.0) or 0.0)
                metadata = getattr(match, "metadata", {}) or {}

                if 0.0 <= raw_score <= 1.0:
                    match_percentage = raw_score * 100.0

                elif -1.0 <= raw_score <= 1.0:
                    match_percentage = max(0.0, raw_score) * 100.0

                else:
                    match_percentage = max(0.0, min(raw_score, 1.0)) * 100.0

                match_percentage = round(match_percentage, 2)

                ranked_results.append(
                    {
                        "resume_id": resume_id,
                        "score": match_percentage,
                        "metadata": metadata,
                    }
                )
            except Exception as error:
                print(f"[ERROR] Failed to process match: {error}")

        ranked_results.sort(key=lambda x: x["score"], reverse=True)

        return ranked_results


