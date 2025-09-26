# Machine Learning Powered Resume Screener

This project provides a light-weight pipeline for screening resume PDFs against a job description. It includes:

- PDF text extraction utilities powered by `pdfminer.six`.
- A simple NLP similarity model based on TF-IDF and cosine similarity.
- A command line interface to rank resume PDFs and export the results.
- Unit tests covering the matching logic.

## Project structure

```
.
├── cli.py                     # Command line entry point
├── resume_screener/
│   ├── __init__.py
│   ├── matching.py            # TF-IDF implementation and similarity scoring
│   ├── pdf_extractor.py       # Helpers to extract text from PDFs
│   ├── pipeline.py            # High-level orchestration helpers
│   └── preprocessing.py       # Tokenisation and text normalisation tools
└── resume_screener/tests/
    └── test_matching.py       # Unit tests
```

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Save your job description to a plain text file, for example `job.txt`.
2. Place all candidate resumes as PDF files in a single directory, for example `resumes/`.
3. Run the CLI to generate the ranking:

```bash
python cli.py job.txt resumes/ --top-k 5 --output results.json
```

The command prints the matches in descending order of similarity and optionally saves them to JSON or CSV based on the output file extension.

## Interactive Streamlit app

To explore the resume screener in the browser, run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

The app allows you to paste or upload a job description (TXT, PDF, or DOCX), add multiple resume PDFs, filter the results, and download the ranked matches as JSON or CSV.

## Running tests

```bash
python -m pytest
```
