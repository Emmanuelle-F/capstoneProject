# from typing import List, Dict, Any
# from utils.embeddings import EmbeddingGenerator
# from utils.pinecone_utils import PineconeManager


# class ResumeRanker:
#     """
#     Handles semantic similarity scoring between resumes and the job description.
#     Steps:
#     - Embed job description
#     - Query Pinecone for nearest resumes
#     - Convert similarity score into a 0–100 percentage
#     - Sort and format results
#     """

#     def __init__(self, index_name: str = "resume-index", embedding_dim: int = 384) -> None:
#         """
#         Initialize embedder and Pinecone manager.

#         Args:
#             index_name (str): Name of Pinecone index.
#             embedding_dim (int): Size of embedding vectors.
#         """
#         self.embedder = EmbeddingGenerator()
#         self.pinecone_manager = PineconeManager(
#             index_name=index_name,
#             dimension=embedding_dim
#         )

#     def rank_resumes(self, job_description: str, top_k: int = 10) -> List[Dict[str, Any]]:
#         """
#         Rank resumes based on semantic similarity.

#         Args:
#             job_description (str): Text of job description.
#             top_k (int): Number of top resumes to return.

#         Returns:
#             List[dict]: List of ranked resumes containing:
#                 - resume_id
#                 - score (0–100)
#                 - metadata (resume text, etc.)
#         """
#         if not job_description or len(job_description.strip()) == 0:
#             raise ValueError("[ERROR] Job description cannot be empty.")

#         # STEP 1 — Embed job description
#         try:
#             job_embedding = self.embedder.embed_text(job_description)
#         except Exception as error:
#             print(f"[ERROR] Failed to embed job description: {error}")
#             return []

#         # STEP 2 — Query Pinecone (returns a QueryResponse object)
#         pinecone_response = self.pinecone_manager.query_similar(
#             query_vector=job_embedding,
#             top_k=top_k,
#         )

#         if not pinecone_response:
#             print("[WARNING] Empty response from Pinecone.")
#             return []

#         # STEP 3 — Access matches directly as an attribute (no dict conversion)
#         matches = getattr(pinecone_response, "matches", None)
#         if not matches:
#             print("[WARNING] No matching resumes found.")
#             return []

#         ranked_results: List[Dict[str, Any]] = []

#         # STEP 4 — Process matches
#         for match in matches:
#             try:
#                 # In Pinecone v3, each match is a ScoredVector object
#                 resume_id = getattr(match, "id", "unknown_id")
#                 raw_score = float(getattr(match, "score", 0.0) or 0.0)  # cosine similarity in [-1, 1]
#                 metadata = getattr(match, "metadata", {}) or {}

#                 # Clamp cosine similarity to [-1, 1] for safety
#                 if raw_score < -1.0:
#                     raw_score = -1.0
#                 elif raw_score > 1.0:
#                     raw_score = 1.0

#                 # Map cosine similarity from [-1, 1] → [0, 100]
#                 # -1 → 0%, 0 → 50%, 1 → 100%
#                 match_percentage = round(((raw_score + 1.0) / 2.0) * 100.0, 2)

#                 ranked_results.append(
#                     {
#                         "resume_id": resume_id,
#                         "score": match_percentage,
#                         "metadata": metadata,
#                     }
#                 )
#             except Exception as error:
#                 print(f"[ERROR] Failed to process match: {error}")

#         # STEP 5 — Sort scores descending
#         ranked_results.sort(key=lambda x: x["score"], reverse=True)

#         return ranked_results

from typing import List, Dict, Any
from utils.embeddings import EmbeddingGenerator
from utils.pinecone_utils import PineconeManager


class ResumeRanker:
    """
    Handles semantic similarity scoring between resumes and the job description.
    Steps:
    - Embed job description
    - Query Pinecone for nearest resumes
    - Convert similarity score into a 0–100 percentage (without inflating 0 -> 50%)
    - Sort and format results
    """

    def __init__(self, index_name: str = "resume-index", embedding_dim: int = 384) -> None:
        """
        Initialize embedder and Pinecone manager.

        Args:
            index_name (str): Name of Pinecone index.
            embedding_dim (int): Size of embedding vectors.
        """
        self.embedder = EmbeddingGenerator()
        self.pinecone_manager = PineconeManager(
            index_name=index_name,
            dimension=embedding_dim
        )

    def rank_resumes(self, job_description: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rank resumes based on semantic similarity.

        Args:
            job_description (str): Text of job description.
            top_k (int): Number of top resumes to return.

        Returns:
            List[dict]: List of ranked resumes containing:
                - resume_id
                - score (0–100)
                - metadata (resume text, etc.)
        """
        if not job_description or len(job_description.strip()) == 0:
            raise ValueError("[ERROR] Job description cannot be empty.")

        # STEP 1 — Embed job description
        try:
            job_embedding = self.embedder.embed_text(job_description)
        except Exception as error:
            print(f"[ERROR] Failed to embed job description: {error}")
            return []

        # STEP 2 — Query Pinecone (returns a QueryResponse object)
        pinecone_response = self.pinecone_manager.query_similar(
            query_vector=job_embedding,
            top_k=top_k,
        )

        if not pinecone_response:
            print("[WARNING] Empty response from Pinecone.")
            return []

        # STEP 3 — Access matches directly as an attribute (no dict conversion)
        matches = getattr(pinecone_response, "matches", None)
        if not matches:
            print("[WARNING] No matching resumes found.")
            return []

        ranked_results: List[Dict[str, Any]] = []

        # STEP 4 — Process matches
        for match in matches:
            try:
                # In Pinecone v3, each match is a ScoredVector object
                resume_id = getattr(match, "id", "unknown_id")
                raw_score = float(getattr(match, "score", 0.0) or 0.0)
                metadata = getattr(match, "metadata", {}) or {}

                # Convert Pinecone score to a human-friendly 0–100 without inflation.
                # Common case: cosine similarity returned in [0, 1] (already non-negative).
                if 0.0 <= raw_score <= 1.0:
                    match_percentage = raw_score * 100.0

                # Less common: cosine similarity in [-1, 1].
                # IMPORTANT: do NOT map 0 -> 50%. Treat 0 as "no similarity".
                elif -1.0 <= raw_score <= 1.0:
                    match_percentage = max(0.0, raw_score) * 100.0

                # Fallback: clamp
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

        # STEP 5 — Sort scores descending
        ranked_results.sort(key=lambda x: x["score"], reverse=True)

        return ranked_results


