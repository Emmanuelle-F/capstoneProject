import os
import tempfile
import re
import html
import hashlib
import io
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils.extract import extract_resume_text
from utils.embeddings import EmbeddingGenerator
from utils.pinecone_utils import PineconeManager
from utils.ranking import ResumeRanker
from utils.summarizer import ResumeSummarizer


# -----------------------------
# Temp file save
# -----------------------------
def save_uploaded_file_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name


# -----------------------------
# Helpers
# -----------------------------
def _extract_candidate_name(resume_text: str) -> str:
    if not resume_text:
        return ""
    for line in resume_text.splitlines():
        line = line.strip()
        if len(line) < 2:
            continue
        if line.lower() in {"resume", "curriculum vitae", "cv"}:
            continue
        return line[:60]
    return ""


def normalize_bullets(text: str, max_items: int = 3) -> str:
    """Force bullets to be newline-separated, even if returned inline."""
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""

    # If bullets exist but are inline, split them
    if "•" in s and "\n" not in s:
        parts = [p.strip() for p in s.split("•") if p.strip()]
    else:
        parts = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if len(parts) <= 1:
            parts = [p.strip() for p in re.split(r"\s-\s+|[.;]\s+", s) if p.strip()]

    cleaned = []
    for p in parts:
        # Remove leading bullet markers or numbering
        p = re.sub(r"^(\s*[-•*]\s+|\s*\d+[\).\]]\s+)", "", p).strip()
        # Remove headings like "Strengths:", "Gaps:", etc.
        p = re.sub(
            r"^(summary|strengths|gaps|red\s*flags|flags)\s*[:\-–]\s*",
            "",
            p,
            flags=re.I,
        ).strip()
        if p:
            cleaned.append(p)

    cleaned = cleaned[:max_items]
    return "\n".join([f"• {c}" for c in cleaned])


def render_multiline_table(df: pd.DataFrame) -> None:
    """HTML table that respects newlines (<br>) + centered headers."""
    df2 = df.copy()
    for col in df2.columns:
        df2[col] = df2[col].astype(str).apply(lambda x: html.escape(x).replace("\n", "<br>"))
    table_html = df2.to_html(index=False, escape=False)

    st.markdown(
        """
        <style>
          .screening-table table { width: 100%; border-collapse: collapse; table-layout: fixed; }
          .screening-table th, .screening-table td {
            border: 1px solid rgba(255,255,255,0.12);
            padding: 10px 12px;
            vertical-align: top;
            font-size: 0.95rem;
            line-height: 1.35;
            overflow-wrap: anywhere;
            word-break: break-word;
          }
          .screening-table th { font-weight: 700; text-align: center; }
          .screening-table td { white-space: normal; }

          /* Column widths: CV, Score, Summary, Strengths, Gaps, Flags */
          .screening-table th:nth-child(1), .screening-table td:nth-child(1) { width: 16%; }
          .screening-table th:nth-child(2), .screening-table td:nth-child(2) { width: 6%;  }
          .screening-table th:nth-child(3), .screening-table td:nth-child(3) { width: 26%; }
          .screening-table th:nth-child(4), .screening-table td:nth-child(4) { width: 17%; }
          .screening-table th:nth-child(5), .screening-table td:nth-child(5) { width: 17%; }
          .screening-table th:nth-child(6), .screening-table td:nth-child(6) { width: 18%; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='screening-table'>{table_html}</div>", unsafe_allow_html=True)


def init_state():
    if "notes_by_cv" not in st.session_state:
        st.session_state["notes_by_cv"] = {}
    if "results_df" not in st.session_state:
        st.session_state["results_df"] = None
    if "compact_df" not in st.session_state:
        st.session_state["compact_df"] = None
    if "last_run_sig" not in st.session_state:
        st.session_state["last_run_sig"] = None


def compute_run_signature(job_description: str, uploaded_files: List[Any]) -> str:
    jd = (job_description or "").strip()
    files_sig: List[Tuple[str, int]] = []
    for f in (uploaded_files or []):
        files_sig.append((getattr(f, "name", "unknown"), int(getattr(f, "size", 0) or 0)))
    files_sig.sort(key=lambda x: (x[0], x[1]))
    raw = jd + "||" + "||".join([f"{n}:{s}" for n, s in files_sig])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def clear_results():
    st.session_state["results_df"] = None
    st.session_state["compact_df"] = None


def render_notes_grid(compact_df: pd.DataFrame) -> None:
    st.markdown("### 📝 Notes")
    st.caption("Add quick a remark for each CV")

    st.markdown(
        """
        <style>
          .notes-header { font-weight: 700; text-align: center; padding: 6px 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    h1, h2, h3 = st.columns([2.5, 1, 6])
    with h1:
        st.markdown("<div class='notes-header'>CV</div>", unsafe_allow_html=True)
    with h2:
        st.markdown("<div class='notes-header'>Score</div>", unsafe_allow_html=True)
    with h3:
        st.markdown("<div class='notes-header'>Notes</div>", unsafe_allow_html=True)

    for _, row in compact_df.iterrows():
        cv_name = str(row["CV"])
        score = str(row["Score"])

        c1, c2, c3 = st.columns([2.5, 1, 6], vertical_alignment="top")
        with c1:
            st.write(cv_name)
        with c2:
            st.write(score)
        with c3:
            key = f"note::{cv_name}"
            default_val = st.session_state["notes_by_cv"].get(cv_name, "")
            note_val = st.text_area(
                label="",
                value=default_val,
                key=key,
                height=60,
                placeholder="Type notes for this CV...",
            )
            st.session_state["notes_by_cv"][cv_name] = note_val


def process_and_index_resumes(
    uploaded_files: List[Any],
    embedder: EmbeddingGenerator,
    pinecone_manager: PineconeManager,
) -> List[Dict[str, Any]]:
    stored_resumes: List[Dict[str, Any]] = []

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        st.write(f"📄 Processing: **{file_name}**")

        temp_path = save_uploaded_file_to_temp(uploaded_file)

        resume_text = extract_resume_text(temp_path)
        if not resume_text:
            st.warning(f"⚠️ Could not extract text from: {file_name}. Skipping.")
            continue

        try:
            resume_embedding = embedder.embed_text(resume_text)
        except Exception as error:
            st.error(f"❌ Failed to embed resume {file_name}: {error}")
            continue

        metadata = {"file_name": file_name, "text": resume_text}

        pinecone_manager.upsert_resume(
            embedding=resume_embedding,
            resume_id=file_name,
            metadata=metadata,
        )

        stored_resumes.append({"resume_id": file_name, "metadata": metadata})

    return stored_resumes

def run_screening(job_description: str, uploaded_files: List[Any]) -> None:
    index_name = "resume-index"
    top_k = 5

    with st.spinner("Getting the AI Assistant ready for this screening..."):
        embedder = EmbeddingGenerator()
        pinecone_manager = PineconeManager(index_name=index_name, dimension=384)

        # 🔴 NEW: clear the index at the beginning of each run
        try:
            pinecone_manager.clear_index()
            # st.info("ℹ️ Cleared existing resumes from Pinecone index for a fresh run.")
        except Exception as e:
            st.warning(f"⚠️ Could not clear Pinecone index: {e}")

        ranker = ResumeRanker(index_name=index_name, embedding_dim=384)
        summarizer = ResumeSummarizer()

    st.success("✅ The AI Assistant is ready to review the resumes.")

    with st.spinner("📄 Reading the resumes and preparing them for comparison..."):
        stored_resumes = process_and_index_resumes(uploaded_files, embedder, pinecone_manager)

    if not stored_resumes:
        st.error("❌ No resumes were processed. Please check your files.")
        clear_results()
        return

    st.success(f"✅ {len(stored_resumes)} resume(s) are ready for screening.")

    with st.spinner("Computing similarity scores and ranking candidates..."):
        ranked_results = ranker.rank_resumes(
            job_description=job_description,
            top_k=min(top_k, len(stored_resumes)),
        )

    if not ranked_results:
        st.warning("⚠️ No matching candidates found.")
        clear_results()
        return

    final_rows = []

    with st.spinner("📝 Generating short summaries for each candidate..."):
        for candidate in ranked_results:
            resume_id = candidate.get("resume_id", "unknown_id")
            score = candidate.get("score", 0.0)
            metadata = candidate.get("metadata", {}) or {}
            resume_text = metadata.get("text", "") or ""

            candidate_name = metadata.get("candidate_name") or _extract_candidate_name(resume_text)

            summary = summarizer.summarize_candidate(
                resume_text=resume_text,
                job_description=job_description,
            )

            strengths_gaps = summarizer.analyze_strengths_and_gaps(
                resume_text=resume_text,
                job_description=job_description,
            )
            strengths_text = strengths_gaps.get("strengths", "")
            gaps_text = strengths_gaps.get("gaps", "")

            red_flags_text = summarizer.analyze_red_flags(
                resume_text=resume_text,
                job_description=job_description,
            )

            final_rows.append(
                {
                    "CV Name": resume_id,
                    "Match Score (%)": round(float(score), 2),
                    "Candidate Name": candidate_name,  # kept internally (not exported)
                    "Summary": summary,
                    "Strengths": strengths_text,
                    "Gaps": gaps_text,
                    "Red Flags": red_flags_text,
                }
            )

    results_df = pd.DataFrame(final_rows)

    compact_df = pd.DataFrame(
        {
            "CV": results_df["CV Name"],
            "Score": results_df["Match Score (%)"].apply(lambda x: f"{float(x):.0f}%"),
            "Summary": results_df["Summary"].apply(lambda x: normalize_bullets(x, max_items=5)),
            "Strengths": results_df["Strengths"].apply(lambda x: normalize_bullets(x, max_items=3)),
            "Gaps": results_df["Gaps"].apply(lambda x: normalize_bullets(x, max_items=3)),
            "Flags": results_df["Red Flags"].apply(lambda x: normalize_bullets(x, max_items=3)),
        }
    )

    try:
        compact_df["_score_num"] = pd.to_numeric(results_df["Match Score (%)"], errors="coerce")
        compact_df = compact_df.sort_values("_score_num", ascending=False).drop(columns=["_score_num"])
    except Exception:
        pass

    st.session_state["results_df"] = results_df
    st.session_state["compact_df"] = compact_df



def make_excel_bytes(results_for_export: pd.DataFrame) -> bytes:
    """
    Create an .xlsx in memory using openpyxl engine.
    If openpyxl isn't installed, raise a friendly error.
    """
    try:
        import openpyxl  # noqa: F401
    except Exception as e:
        raise RuntimeError("Excel export requires 'openpyxl'. Install it with: pip install openpyxl") from e

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results_for_export.to_excel(writer, index=False, sheet_name="Screening Results")
        worksheet = writer.sheets["Screening Results"]
        worksheet.freeze_panes = "A2"  # freeze header row
    output.seek(0)
    return output.getvalue()


def render_dashboard_charts(results_df: pd.DataFrame, top_n: int = 5) -> None:
    """Show only a compact Top candidates bar chart in the center of the page."""
    if results_df is None or results_df.empty or "Match Score (%)" not in results_df.columns:
        st.info("No results available to display charts.")
        return

    df = results_df.copy()
    df["score_num"] = pd.to_numeric(df["Match Score (%)"], errors="coerce")
    df = df.dropna(subset=["score_num"])
    if df.empty:
        st.info("No valid numeric scores found to display charts.")
        return

    st.markdown("### 📈 Top Candidates")

    top = df.sort_values("score_num", ascending=False).head(top_n)
    if top.empty:
        st.info("No data available for top candidates chart.")
        return

    # Create 3 columns and use only the middle one to keep the chart narrow
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        # Smaller figure size so it does not feel huge
        fig, ax = plt.subplots(figsize=(6, 3))  # width, height in inches

        ax.barh(top["CV Name"], top["score_num"])
        ax.invert_yaxis()

        for i, (score, cv_name) in enumerate(zip(top["score_num"], top["CV Name"])):  # noqa: B007
            ax.text(
                score + 1,
                i,
                f"{score:.0f}%",
                va="center",
                fontsize=8,
            )

        ax.set_title(f"Top {len(top)} Candidates by Match Score", fontsize=10)
        ax.set_xlabel("Match Score (%)", fontsize=9)
        ax.set_ylabel("")
        ax.set_xlim(0, 100)

        ax.tick_params(axis="both", labelsize=8)

        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)


def main():
    # Use a very safe icon for the browser tab, keep robot in title
    st.set_page_config(page_title="AI Resume Assistant", page_icon="🤖", layout="wide")
    init_state()

    st.title("🧠 AI Resume Assistant")
    st.caption("The AI Assistant that helps you spot the right talent faster")
    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("1️⃣ Input the Job Description")
        job_description = st.text_area(
            "Input the job description here:",
            height=220,
            key="jd_input",
            placeholder="Example: We are looking for a Software Engineer with strong skills in Python, REST APIs...",
        )

    with col_right:
        st.subheader("2️⃣ Upload Resumes")
        uploaded_files = st.file_uploader(
            "Upload one or more resumes (PDF or DOCX only):",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="resume_uploader",
        )

    current_sig = compute_run_signature(job_description, uploaded_files)
    if st.session_state["last_run_sig"] is not None and st.session_state["last_run_sig"] != current_sig:
        clear_results()

    st.markdown("---")
    run_button = st.button("🚀 Submit")

    if run_button:
        if not job_description or len(job_description.strip()) == 0:
            st.error("❌ Please input a job description before submiting.")
        elif not uploaded_files:
            st.error("❌ Please upload at least one resume.")
        else:
            st.session_state["last_run_sig"] = current_sig
            run_screening(job_description, uploaded_files)

    if st.session_state["compact_df"] is not None and st.session_state["results_df"] is not None:
        st.markdown("### 📊 Results")
        render_multiline_table(st.session_state["compact_df"])

        # ✅ Dashboard charts (Histogram + Top candidates)
        render_dashboard_charts(st.session_state["results_df"], top_n=5)

        # Notes (inline)
        render_notes_grid(st.session_state["compact_df"])

        # Build export dataframe WITHOUT Candidate Name (remove Excel column C)
        notes_map = st.session_state.get("notes_by_cv", {})
        results_for_export = st.session_state["results_df"].copy()

        if "Candidate Name" in results_for_export.columns:
            results_for_export = results_for_export.drop(columns=["Candidate Name"])

        results_for_export["Notes"] = results_for_export["CV Name"].map(lambda cv: notes_map.get(str(cv), ""))

        try:
            excel_bytes = make_excel_bytes(results_for_export)
            st.download_button(
                label="📥 Download Results",
                data=excel_bytes,
                file_name="AI_Resume_Assistant_Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(str(e))


if __name__ == "__main__":
    main()
