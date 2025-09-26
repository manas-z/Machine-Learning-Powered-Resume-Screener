"""Utilities for extracting text from resume PDF files."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union


class PDFExtractionError(RuntimeError):
    """Raised when a PDF file cannot be processed."""


def _coerce_path(path: Union[str, Path]) -> Path:
    if isinstance(path, Path):
        return path
    return Path(path)


def extract_text_from_pdf(path: Union[str, Path]) -> str:
    """Extract raw text from a PDF file.

    Parameters
    ----------
    path:
        Path to the PDF file. Strings are automatically converted to
        :class:`~pathlib.Path` objects.

    Returns
    -------
    str
        The extracted text.

    Raises
    ------
    PDFExtractionError
        If the file does not exist or cannot be parsed.
    """

    pdf_path = _coerce_path(path)
    if not pdf_path.exists():
        raise PDFExtractionError(f"PDF file not found: {pdf_path}")

    try:
        from pdfminer.high_level import extract_text  # type: ignore
    except Exception as exc:  # pragma: no cover - import error is environment specific
        raise PDFExtractionError(
            "Failed to import pdfminer.six. Install it with 'pip install pdfminer.six'."
        ) from exc

    try:
        text = extract_text(str(pdf_path))
    except Exception as exc:  # pragma: no cover - pdfminer errors difficult to trigger reliably
        raise PDFExtractionError(f"Failed to extract text from {pdf_path}") from exc

    if not text.strip():
        raise PDFExtractionError(f"No text could be extracted from {pdf_path}")

    return text


def batch_extract_text(paths: Iterable[Union[str, Path]]) -> dict[str, str]:
    """Extract text from multiple PDF files.

    Parameters
    ----------
    paths:
        Iterable of PDF file paths.

    Returns
    -------
    dict[str, str]
        Mapping of file names to extracted text.
    """

    results: dict[str, str] = {}
    for path in paths:
        pdf_path = _coerce_path(path)
        results[pdf_path.stem] = extract_text_from_pdf(pdf_path)
    return results
