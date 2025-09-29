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

try:
    from docx import Document
except ModuleNotFoundError:  # pragma: no cover - optional dependency for typing only
    Document = None  # type: ignore[assignment]


class JobDescriptionExtractionError(RuntimeError):
    """Raised when the job description file cannot be processed."""


def _extract_resume_texts(uploaded_files: Iterable[UploadedFile]) -> dict[str, str]:
    """Convert uploaded PDFs into a mapping of resume ids to raw text."""

    resume_texts: dict[str, str] = {}
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        resume_stem = Path(uploaded_file.name).stem or f"resume_{index}"
        try:
            resume_texts[resume_stem] = _extract_pdf_text(uploaded_file)
        except PDFExtractionError as exc:
            raise PDFExtractionError(f"Failed to read '{uploaded_file.name}': {exc}") from exc

    return resume_texts


def _extract_pdf_text(uploaded_file: UploadedFile) -> str:
    """Extract the text content from an uploaded PDF file."""

    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.getbuffer())
        temp_path = Path(temp_pdf.name)

    try:
        return extract_text_from_pdf(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _extract_job_description(
    text_input: str, uploaded_file: UploadedFile | None
) -> str:
    """Return the job description from manual text or an uploaded document."""

    if uploaded_file is None:
        return text_input

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf_text(uploaded_file)

    if suffix in {".docx"}:
        if Document is None:  # pragma: no cover - dependency guard
            raise JobDescriptionExtractionError(
                "python-docx is required to read Word documents. Please install the optional dependency."
            )
        document = Document(io.BytesIO(uploaded_file.getbuffer()))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    if suffix == ".txt":
        try:
            return uploaded_file.getvalue().decode("utf-8")
        except UnicodeDecodeError as exc:  # pragma: no cover - rare encoding issue
            raise JobDescriptionExtractionError(
                "Could not decode the uploaded text file. Please ensure it is UTF-8 encoded."
            ) from exc

    raise JobDescriptionExtractionError(
        "Unsupported file type. Please upload a PDF, DOCX, or TXT file."
    )


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
        "Paste your job description, upload a PDF/DOCX/TXT file, upload resume PDFs, and compare them using a TF-IDF cosine similarity score."
    )

    job_description_input = st.text_area("Job description", height=220)
    job_description_file = st.file_uploader(
        "Job description file (optional)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=False,
        key="job_description_file",
    )
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

    try:
        job_description = _extract_job_description(job_description_input, job_description_file)
    except (PDFExtractionError, JobDescriptionExtractionError) as exc:
        st.error(str(exc))
        return

    if not job_description.strip():
        st.error("Please provide a job description by typing it or uploading a file before running the screener.")
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
