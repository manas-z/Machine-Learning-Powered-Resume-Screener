"""Streamlit interface for the simple resume screener."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from resume_screener import matching
from resume_screener.pdf_extractor import PDFExtractionError, extract_text_from_pdf


def _extract_resume_texts(uploaded_files: Iterable[UploadedFile]) -> dict[str, str]:
    """Convert uploaded PDFs into a mapping of resume ids to raw text."""

    resume_texts: dict[str, str] = {}
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        resume_stem = Path(uploaded_file.name).stem or f"resume_{index}"
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.getbuffer())
            temp_path = Path(temp_pdf.name)

        try:
            resume_texts[resume_stem] = extract_text_from_pdf(temp_path)
        except PDFExtractionError as exc:
            raise PDFExtractionError(f"Failed to read '{uploaded_file.name}': {exc}") from exc
        finally:
            temp_path.unlink(missing_ok=True)

    return resume_texts


def _download_button(matches: list[matching.ResumeMatch]) -> None:
    """Offer a CSV download for the screening results."""

    summary = matching.summarise_matches(matches)
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=["resume_id", "score", "keywords"])
    writer.writeheader()
    writer.writerows(summary)
    st.download_button(
        label="Download CSV",
        data=csv_buffer.getvalue().encode("utf-8"),
        file_name="resume_matches.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Simple Resume Screener", page_icon="📄")
    st.title("Simple Resume Screener")
    st.write(
        "Paste your job description, upload resume PDFs, and compare them using a TF-IDF cosine similarity score."
    )

    job_description = st.text_area("Job description", height=220)
    uploaded_resumes = st.file_uploader(
        "Resume PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        min_score = st.slider("Minimum score", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    with col2:
        top_k_input = st.number_input("Top K results (0 = show all)", min_value=0, value=0, step=1)

    run_clicked = st.button("Run screener", type="primary")

    if not run_clicked:
        return

    if not job_description.strip():
        st.error("Please provide a job description before running the screener.")
        return

    if not uploaded_resumes:
        st.error("Upload at least one resume PDF to continue.")
        return

    try:
        resume_texts = _extract_resume_texts(uploaded_resumes)
    except PDFExtractionError as exc:
        st.error(str(exc))
        return

    top_k = int(top_k_input) or None
    matches = matching.score_resumes(
        job_description,
        resume_texts,
        top_k=top_k,
        min_score=min_score,
    )

    if not matches:
        st.warning("No resumes passed the similarity threshold.")
        return

    rows = [
        {
            "Resume": match.resume_id,
            "Score": round(match.score, 3),
            "Keywords": ", ".join(match.keywords),
        }
        for match in matches
    ]

    st.dataframe(rows, use_container_width=True)
    _download_button(matches)

    st.caption(
        "Scores range from 0 to 1 and represent cosine similarity between TF-IDF vectors of the job description and each resume."
    )


if __name__ == "__main__":  # pragma: no cover - Streamlit entry point
    main()
