"""Streamlit application for the resume screening pipeline."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from docx import Document



from resume_screener import pipeline
from resume_screener.matching import ResumeMatch
from resume_screener.pdf_extractor import PDFExtractionError, extract_text_from_pdf


def _extract_resume_texts(uploaded_files: Iterable[UploadedFile]) -> dict[str, str]:
    """Convert uploaded PDF files into a mapping of resume ids to raw text."""

    resume_texts: dict[str, str] = {}
    for uploaded_file in uploaded_files:
        resume_name = Path(uploaded_file.name).stem or "resume"
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            uploaded_file.seek(0)
            temp_pdf.write(uploaded_file.read())
            temp_path = Path(temp_pdf.name)

        try:
            resume_texts[resume_name] = extract_text_from_pdf(temp_path)
        except PDFExtractionError as exc:
            raise PDFExtractionError(f"Failed to process '{uploaded_file.name}': {exc}") from exc
        finally:
            temp_path.unlink(missing_ok=True)

    return resume_texts


def _load_job_description(uploaded_file: UploadedFile) -> str:
    """Extract text from an uploaded job description file."""

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".txt":
        uploaded_file.seek(0)
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            uploaded_file.seek(0)
            temp_pdf.write(uploaded_file.read())
            temp_path = Path(temp_pdf.name)

        try:
            return extract_text_from_pdf(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    if suffix == ".docx":
        uploaded_file.seek(0)
        document = Document(io.BytesIO(uploaded_file.read()))
        text_blocks: list[str] = [
            paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()
        ]
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if cell_text:
                        text_blocks.append(cell_text)
        return "\n".join(text_blocks)

    raise ValueError("Unsupported job description file type.")


def _render_download_buttons(matches: list[ResumeMatch]) -> None:
    summary = pipeline.summarise_matches(matches)

    json_bytes = json.dumps(summary, indent=2).encode("utf-8")
    st.download_button(
        "Download results as JSON",
        data=json_bytes,
        file_name="resume_matches.json",
        mime="application/json",
    )

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=["resume_id", "score", "highlights"])
    writer.writeheader()
    writer.writerows(summary)
    st.download_button(
        "Download results as CSV",
        data=csv_buffer.getvalue().encode("utf-8"),
        file_name="resume_matches.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Resume Screener", page_icon="📄", layout="wide")
    st.title("📄 Machine Learning Powered Resume Screener")
    st.markdown(
        "Use natural language processing to compare candidate resumes with your job description. "
        "Upload a job description, add PDF resumes, and review the ranked matches complete with highlighted keywords."
    )

    with st.sidebar:
        st.header("Configuration")
        top_k_option = st.number_input(
            "Number of top matches",
            min_value=1,
            value=5,
            step=1,
            help="Limit the number of resumes returned. Disable this below to keep all matches.",
        )
        min_score = st.slider(
            "Minimum similarity score",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Filter out resumes whose cosine similarity is below this threshold.",
        )
        keep_all = st.checkbox("Return all matches", value=True)
        top_k = None if keep_all else int(top_k_option)

    st.subheader("Job description")
    job_description_file = st.file_uploader(
        "Upload a job description file (optional)",
        type=["txt", "pdf", "docx"],
        help=(
            "Upload a job description document or leave this blank and type the "
            "details manually in the editor below."
        ),
    )

    initial_job_description = ""
    if job_description_file is not None:
        try:
            initial_job_description = _load_job_description(job_description_file)
        except (ValueError, PDFExtractionError) as exc:
            st.error(str(exc))
            return


    job_description_text = st.text_area(
        "Paste or edit the job description",
        value=initial_job_description,
        height=200,
        placeholder="Describe the responsibilities, required skills, and experience for the role...",
    )

    st.subheader("Candidate resumes")
    resume_files = st.file_uploader(
        "Upload resume PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files to screen against the job description.",
    )

    if st.button("Screen resumes", type="primary"):
        if not job_description_text.strip():
            st.error("Please provide a job description before screening resumes.")
            return
        if not resume_files:
            st.error("Please upload at least one resume in PDF format.")
            return

        with st.spinner("Extracting text and ranking resumes..."):
            try:
                resume_texts = _extract_resume_texts(resume_files)
            except PDFExtractionError as exc:
                st.error(str(exc))
                return

            try:
                matches = pipeline.screen_resumes(
                    job_description_text,
                    resume_texts,
                    top_k=top_k,
                    min_score=min_score,
                )
            except ValueError as exc:
                st.error(str(exc))
                return

        if not matches:
            st.warning(
                "No resumes met the criteria. Try lowering the minimum similarity score or adding more resumes."
            )
            return

        st.success(f"Found {len(matches)} matching resume{'s' if len(matches) != 1 else ''}.")

        table_data = [
            {
                "Resume": match.resume_id,
                "Score": round(match.score, 3),
                "Highlights": ", ".join(match.highlights),
            }
            for match in matches
        ]
        st.dataframe(table_data, use_container_width=True)

        st.markdown("### Export results")
        _render_download_buttons(matches)

        st.caption(
            "Similarity scores are based on TF-IDF cosine similarity between the job description and resume content."
        )


if __name__ == "__main__":
    main()
