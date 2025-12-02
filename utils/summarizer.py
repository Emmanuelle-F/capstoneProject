import os
from typing import Dict

from dotenv import load_dotenv
from llama_index.llms.groq import Groq

# Load environment variables from .env
load_dotenv()


class ResumeSummarizer:
    """
    Uses a Groq-hosted LLM to generate readable summaries
    for each candidate based on their resume and the job description.
    """

    def __init__(self) -> None:
        """
        Initialize the Groq LLM client using settings from environment variables.
        """
        api_key = os.getenv("GROQ_API_KEY")
        model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

        if not api_key:
            raise ValueError(
                "[ERROR] GROQ_API_KEY is missing in .env. "
                "Please set it before using ResumeSummarizer."
            )

        try:
            self.llm = Groq(
                model=model_name,
                api_key=api_key,
            )
            print(f"[INFO] Groq LLM initialized with model: {model_name}")
        except Exception as error:
            raise RuntimeError(f"[ERROR] Failed to initialize Groq LLM: {error}")

    def summarize_candidate(self, resume_text: str, job_description: str) -> str:
        """
        Generate a concise, recruiter-friendly summary for a candidate.

        Args:
            resume_text (str): Full text extracted from the candidate's resume.
            job_description (str): The target job description text.

        Returns:
            str: A short markdown-style summary describing fit, strengths, and gaps.
        """
        if not resume_text or len(resume_text.strip()) == 0:
            return "[WARNING] Empty resume text. Cannot generate summary."

        if not job_description or len(job_description.strip()) == 0:
            return "[WARNING] Empty job description. Cannot generate summary."

        prompt = f"""
        You are an assistant helping an HR recruiter evaluate candidates.

        You are given:
        1) A JOB DESCRIPTION
        2) A CANDIDATE RESUME

        TASK:
        - Write a concise, to the point and objective summary (5 bullets points ONLY) of the candidate.
        - Focus on their qualifications, experience, key skills, and seniority.
        - Use a neutral, professional tone.
        - Each bullet should be 1 line, direct and to the point.
        - Do not write sentances but meaningful note points, not over 10 words

        JOB DESCRIPTION:
        \"\"\"{job_description}\"\"\"

        CANDIDATE RESUME:
        \"\"\"{resume_text}\"\"\"

        Now write the summary.
        """

        try:
            response = self.llm.complete(prompt)
            summary_text = response.text.strip()
            return summary_text
        except Exception as error:
            print(f"[ERROR] Failed to generate summary with Groq LLM: {error}")
            return "[ERROR] Could not generate summary due to an LLM error."
        

    def analyze_strengths_and_gaps(self, resume_text: str, job_description: str) -> dict:
        """
        Analyse a candidate's strengths and gaps relative to the job description.

        Returns a dict with:
            {
                "strengths": str,
                "gaps": str
            }
        Both fields are short bullet lists in plain text.
        """
        if not resume_text or not resume_text.strip():
            return {
                "strengths": "[WARNING] Empty resume text. Cannot analyse strengths.",
                "gaps": "[WARNING] Empty resume text. Cannot analyse gaps.",
            }

        if not job_description or not job_description.strip():
            return {
                "strengths": "[WARNING] Empty job description. Cannot analyse strengths.",
                "gaps": "[WARNING] Empty job description. Cannot analyse gaps.",
            }

        prompt = f"""
        You are assisting an HR recruiter.

        You are given:
        1) A JOB DESCRIPTION
        2) A CANDIDATE RESUME

        STRICT OUTPUT FORMAT (follow exactly, including headings and dashes):

        Strengths:
         <strength 1>
         <strength 2>
         <strength 3>

        Gaps:
         <gap 1>
         <gap 2>
         <gap 3>

        RULES:
        - Each bullet MUST be on its own line (newline separated). No inline bullets like "• a • b • c".
        - Keep each bullet short: max 10 words.
        - Strengths: only job-relevant strengths supported by the resume.
        - Gaps: only missing requirements vs the job description (skills, tools, years, domain).
        - Do NOT add extra sections, explanations, or summaries.
        - If no clear gaps, write exactly:
        - No major gaps identified
        - No major gaps identified
        - No major gaps identified

        JOB DESCRIPTION:
        \"\"\"{job_description}\"\"\"

        CANDIDATE RESUME:
        \"\"\"{resume_text}\"\"\"
        """

        try:
            response = self.llm.complete(prompt)
            full_text = response.text.strip()

            # Simple split based on "Gaps:" label
            strengths_part = ""
            gaps_part = ""

            if "Gaps:" in full_text:
                parts = full_text.split("Gaps:", 1)
                strengths_part = parts[0].strip()
                gaps_part = "Gaps:" + parts[1].strip()
            else:
                strengths_part = full_text
                gaps_part = "[INFO] No explicit gaps section returned by the model."

            return {
                "strengths": strengths_part,
                "gaps": gaps_part,
            }

        except Exception as error:
            print(f"[ERROR] Failed to analyse strengths/gaps: {error}")
            return {
                "strengths": "[ERROR] Could not generate strengths due to an LLM error.",
                "gaps": "[ERROR] Could not generate gaps due to an LLM error.",
            }
        

    def analyze_red_flags(self, resume_text: str, job_description: str) -> str:
        """
        Identify potential risk factors or points to clarify for this candidate.

        Returns:
            str: A short list of bullet points, or a message saying no obvious red flags.
        """
        if not resume_text or not resume_text.strip():
            return "[WARNING] Empty resume text. Cannot analyse red flags."

        if not job_description or not job_description.strip():
            return "[WARNING] Empty job description. Cannot analyse red flags."

        prompt = f"""
        You are an assistant helping an HR recruiter evaluate candidates.

        Given the JOB DESCRIPTION and CANDIDATE RESUME below, list any
        potential risk factors or red flags.

        Focus ONLY on the following points:
        - Very frequent job changes ONLY when they occur
        within a short period (e.g. many roles of LESS THAN 1 year each in the LAST 5 years).
        - Do NOT consider it a red flag if the candidate has a long career
        (e.g. 10–15 years) with only a few roles and normal progression
        (e.g. 2–4 jobs over that period).
        - Limited industry experience applies ONLY if the candidate has LESS THAN 2 years of working experience in total
        - Long unexplained employment gaps.

        If there are no major concerns, say clearly:
        "No obvious red flags; profile appears generally consistent."

        - Output 3 bullet points and be concise.
        - Each bullet should be 1 line, direct and to the point.
        - Do not write sentances but meaningful note points, not over 10 words

        JOB DESCRIPTION:
        \"\"\"{job_description}\"\"\"

        CANDIDATE RESUME:
        \"\"\"{resume_text}\"\"\"
        """

        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as error:
            print(f"[ERROR] Failed to analyse red flags: {error}")
            return "[ERROR] Could not generate red flag analysis due to an LLM error."


        

        