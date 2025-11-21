# from utils.extract import extract_resume_text

# print(extract_resume_text(r"C:\Users\emmaf\Documents\AI_Resume_Assistant\data\resume\Emmanuelle Frappier CV - EN.pdf"))

# print(extract_resume_text(r"C:\Users\emmaf\Documents\AI_Resume_Assistant\data\resume\Emmanuelle Frappier CV - EN.docx"))

########################################################################################################################################

# from utils.embeddings import EmbeddingGenerator

# texts = [
#     "Python developer with machine learning skills.",
#     "Front-end engineer skilled in React.",
#     "Financial accountant with tax experience."
# ]

# embedder = EmbeddingGenerator()
# vectors = embedder.embed_multiple(texts)

# for i, vec in enumerate(vectors):
#     print(f"Vector {i} shape:", vec.shape)

########################################################################################################################################

# from utils.pinecone_utils import PineconeManager
# import numpy as np

# def test_pinecone():
#     print("----- PINECONE TEST STARTED -----")

#     # 1. Initialize Pinecone manager
#     try:
#         pinecone_manager = PineconeManager(index_name="resume-index", dimension=384)
#         print("[TEST] Pinecone initialization ✔")
#     except Exception as e:
#         print("[FAILED] Could not initialize Pinecone:", e)
#         return

#     # 2. Create a fake embedding vector
#     test_vector = np.random.rand(384)  # 384-dim vector
#     resume_id = "test_resume"
#     metadata = {"text": "This is a test resume."}

#     # 3. Upsert the test embedding
#     try:
#         pinecone_manager.upsert_resume(test_vector, resume_id, metadata)
#         print("[TEST] Upsert successful ✔")
#     except Exception as e:
#         print("[FAILED] Upsert error:", e)
#         return

#     # 4. Query using the same vector
#     try:
#         response = pinecone_manager.query_similar(test_vector, top_k=3)
#         print("[TEST] Query successful ✔")
#         print("--- Pinecone Query Response ---")
#         print(response)
#     except Exception as e:
#         print("[FAILED] Query failed:", e)
#         return

#     print("----- PINECONE TEST COMPLETED ✔ -----")

# if __name__ == "__main__":
#     test_pinecone()

########################################################################################################################################

import os
from typing import List

from utils.extract import extract_resume_text
from utils.embeddings import EmbeddingGenerator
from utils.pinecone_utils import PineconeManager
from utils.ranking import ResumeRanker


def get_resume_files(resume_folder: str) -> List[str]:
    """
    List all PDF and DOCX resumes in the given folder.
    """
    supported_extensions = (".pdf", ".docx")
    files = []

    for file_name in os.listdir(resume_folder):
        if file_name.lower().endswith(supported_extensions):
            files.append(os.path.join(resume_folder, file_name))

    return files


def index_sample_resumes(resume_folder: str, index_name: str = "resume-index") -> None:
    """
    Extract text from resumes, generate embeddings, and upsert them into Pinecone.
    """
    resume_files = get_resume_files(resume_folder)

    if not resume_files:
        print(f"[WARNING] No PDF/DOCX files found in: {resume_folder}")
        return

    print(f"[INFO] Found {len(resume_files)} resume(s) to index.")

    embedder = EmbeddingGenerator()
    pinecone_manager = PineconeManager(index_name=index_name, dimension=384)

    for resume_path in resume_files:
        file_name = os.path.basename(resume_path)
        print(f"[INFO] Processing resume: {file_name}")

        # 1) Extract text
        resume_text = extract_resume_text(resume_path)
        if not resume_text:
            print(f"[WARNING] Empty text for: {file_name}, skipping.")
            continue

        # 2) Generate embedding
        try:
            resume_embedding = embedder.embed_text(resume_text)
        except Exception as error:
            print(f"[ERROR] Failed to embed resume {file_name}: {error}")
            continue

        # 3) Upsert into Pinecone
        metadata = {
            "file_name": file_name,
            "text": resume_text
        }

        pinecone_manager.upsert_resume(
            embedding=resume_embedding,
            resume_id=file_name,
            metadata=metadata
        )


def test_ranking_flow():
    """
    Full test:
    - Index sample resumes into Pinecone
    - Rank them against a job description
    - Print match scores
    """
    resume_folder = "data/resume"
    index_name = "resume-index"

    # STEP 1: Index resumes (only needs to be done once, but safe to re-run)
    index_sample_resumes(resume_folder=resume_folder, index_name=index_name)

    # STEP 2: Define a test job description
    job_description = """
    We are looking for a Software Engineer with strong skills in Python, REST APIs,
    and cloud services (AWS or Azure). Experience with backend development, databases,
    and version control (Git) is required. Knowledge of Docker is a plus.
    """

    # STEP 3: Rank resumes
    ranker = ResumeRanker(index_name=index_name, embedding_dim=384)
    ranked_results = ranker.rank_resumes(job_description=job_description, top_k=5)

    print("\n----- RANKING RESULTS -----\n")

    if not ranked_results:
        print("No results returned. Check if resumes were indexed correctly.")
        return

    for result in ranked_results:
        resume_id = result["resume_id"]
        score = result["score"]
        metadata = result.get("metadata", {})

        print(f"Resume: {resume_id}")
        print(f"Match Score: {score}%")

        # Print a short preview of the resume text (first 200 chars)
        text_preview = metadata.get("text", "")[:200]
        print(f"Text Preview: {text_preview}...")
        print("-" * 60)


if __name__ == "__main__":
    test_ranking_flow()

