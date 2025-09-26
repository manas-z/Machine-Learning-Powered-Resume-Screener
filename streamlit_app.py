"""Streamlit application for the resume screening pipeline."""
from __future__ import annotations

import csv
import io
import json
from statistics import mean
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile



from resume_screener import matching, pipeline, preprocessing
from resume_screener.matching import KeywordInsight, ResumeMatch
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
        from docx import Document

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


def _format_keyword_list(keywords: Iterable[str]) -> str:
    items = sorted({keyword for keyword in keywords if keyword})
    return ", ".join(items) if items else "—"


def _keyword_coverage_summary(
    insights: list[KeywordInsight], resume_tokens: dict[str, tuple[str, ...]]
) -> dict[str, tuple[float, list[str]]]:
    top_terms = [insight.term for insight in insights]
    if not top_terms:
        return {}

    coverage: dict[str, tuple[float, list[str]]] = {}
    top_term_count = len(top_terms)
    for resume_id, tokens in resume_tokens.items():
        token_set = set(tokens)
        matched = [term for term in top_terms if term in token_set]
        ratio = len(matched) / top_term_count if top_term_count else 0.0
        coverage[resume_id] = (ratio, matched)
    return coverage


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

    uploaded_job_description_text = ""
    if job_description_file is not None:
        try:
            uploaded_job_description_text = _load_job_description(job_description_file)
        except (ValueError, PDFExtractionError) as exc:
            st.error(str(exc))
            return

    input_mode = st.radio(
        "How would you like to provide the job description?",
        (
            "Use uploaded document",
            "Type manually",
        ),
        index=0 if job_description_file is not None else 1,
        help="Choose whether to rely on the uploaded document or enter the description directly.",
    )
    job_description_from_upload = uploaded_job_description_text
    manual_job_description = ""
    if input_mode == "Use uploaded document" and job_description_file is not None:
        job_description_from_upload = st.text_area(
            "Review and optionally edit the job description",
            value=uploaded_job_description_text,
            key="uploaded_job_description",
            height=240,
        )
    elif input_mode == "Type manually":
        manual_job_description = st.text_area(
            "Paste or type the job description",
            key="manual_job_description",
            height=240,
            placeholder=(
                "Describe the responsibilities, required skills, and experience for the role..."
            ),
        )
    else:
        st.info("Upload a job description document or switch to manual entry to continue.")

    st.divider()
    st.subheader("Keyword strategy")
    required_keywords_input = st.text_input(
        "Must-have keywords (comma separated)",
        help=(
            "Ensure critical skills are surfaced. The screener highlights resumes missing these keywords and "
            "can optionally remove them from the final results."
        ),
    )
    enforce_keywords = st.checkbox(
        "Hide resumes missing any must-have keyword",
        value=False,
    )

    st.subheader("Candidate resumes")
    resume_files = st.file_uploader(
        "Upload resume PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files to screen against the job description.",
    )

    if st.button("Screen resumes", type="primary"):
        job_description_text = ""
        if input_mode == "Use uploaded document":
            if job_description_file is None:
                st.error("Please upload a job description document or switch to manual entry.")
                return
            job_description_text = job_description_from_upload
        else:
            job_description_text = manual_job_description

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

        try:
            corpus = matching.prepare_corpus(job_description_text, resume_texts)
        except ValueError as exc:
            st.error(str(exc))
            return

        required_tokens = preprocessing.tokenize(required_keywords_input)
        keyword_insights = matching.extract_keyword_insights(corpus, limit=15)
        coverage_summary = _keyword_coverage_summary(keyword_insights, corpus.resume_tokens)

        filtered_matches: list[ResumeMatch] = []
        missing_required: dict[str, list[str]] = {}
        dropped_resumes = []
        for match in matches:
            resume_terms = set(corpus.resume_tokens.get(match.resume_id, ()))
            missing = [token for token in required_tokens if token not in resume_terms]
            missing_required[match.resume_id] = missing
            if enforce_keywords and missing:
                dropped_resumes.append(match.resume_id)
                continue
            filtered_matches.append(match)

        if enforce_keywords and dropped_resumes:
            st.info(
                "Filtered out resumes missing must-have keywords: "
                + ", ".join(sorted(dropped_resumes))
            )

        if not filtered_matches:
            st.warning("No resumes satisfied the configured keyword and score filters.")
            return

        matches = filtered_matches

        st.success(f"Found {len(matches)} matching resume{'s' if len(matches) != 1 else ''}.")

        match_scores = [match.score for match in matches]
        score_high = max(match_scores)
        score_avg = mean(match_scores) if match_scores else 0.0

        st.markdown("### Screening summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Resumes processed", len(resume_texts))
        col2.metric("Matches returned", len(matches))
        col3.metric("Highest score", f"{score_high:.0%}")
        col4.metric("Average score", f"{score_avg:.0%}")

        tabs = st.tabs(["Ranked results", "Keyword coverage", "Candidate breakdown"])

        table_data = []
        for match in matches:
            coverage_ratio, matched_keywords = coverage_summary.get(match.resume_id, (0.0, []))
            table_data.append(
                {
                    "Resume": match.resume_id,
                    "Score": round(match.score, 3),
                    "Keyword coverage": f"{int(coverage_ratio * 100)}%",
                    "Highlights": _format_keyword_list(match.highlights),
                    "Missing must-have keywords": _format_keyword_list(missing_required.get(match.resume_id, [])),
                }
            )

        with tabs[0]:
            st.dataframe(table_data, use_container_width=True)
            st.markdown("### Export results")
            _render_download_buttons(matches)

        with tabs[1]:
            if not keyword_insights:
                st.info("Keyword coverage becomes available once there are meaningful job description terms.")
            else:
                coverage_table = [
                    {
                        "Keyword": insight.term,
                        "Importance": round(insight.weight, 3),
                        "Resumes containing": insight.resume_count,
                        "Coverage": f"{insight.coverage_ratio * 100:.0f}%",
                    }
                    for insight in keyword_insights
                ]
                st.dataframe(coverage_table, use_container_width=True)

        with tabs[2]:
            if not keyword_insights:
                st.info("Run the screener to generate candidate insights.")
            for match in matches:
                coverage_ratio, matched_keywords = coverage_summary.get(match.resume_id, (0.0, []))
                missing_keywords = matching.missing_keywords_for_resume(corpus, match.resume_id, top_n=10)
                with st.expander(f"{match.resume_id} — Score {match.score:.0%}"):
                    st.write("Keyword alignment")
                    st.progress(coverage_ratio, text=f"{int(coverage_ratio * 100)}% of top job keywords matched")
                    st.write("Matched keywords:")
                    st.caption(_format_keyword_list(matched_keywords))
                    st.write("Top missing job keywords:")
                    st.caption(_format_keyword_list(missing_keywords))
                    missing_required_keywords = missing_required.get(match.resume_id, [])
                    if missing_required_keywords:
                        st.warning(
                            "Missing must-have keywords: "
                            + ", ".join(sorted(missing_required_keywords))
                        )
                    st.write("Highlights from the resume:")
                    st.caption(_format_keyword_list(match.highlights))

        st.caption(
            "Similarity scores are based on TF-IDF cosine similarity between the job description and resume content."
        )


if __name__ == "__main__":
    main()
