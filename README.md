# Simple Resume Screener

This project showcases a lightweight resume screener that uses TF-IDF and cosine similarity to compare resume PDFs with a job description. It is intentionally focused on the fundamentals so it is easy to explain in a portfolio or demo setting.

## Features

- Extract text from resume PDFs with `pdfminer.six`.
- Score resumes against a job description using TF-IDF cosine similarity.
- Simple Streamlit interface for quick experimentation.

## Project structure

```
.
├── resume_screener/
│   ├── __init__.py
│   ├── matching.py            # TF-IDF scoring helper
│   ├── pdf_extractor.py       # PDF text extraction utilities
│   ├── pipeline.py            # Small orchestration helpers
│   └── preprocessing.py       # Tokenisation helper used for keyword highlights
└── streamlit_app.py           # Minimal Streamlit UI
```

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Streamlit app

Launch the interactive interface with:

```bash
streamlit run streamlit_app.py
```

Paste your job description, upload PDFs, and view the similarity scores directly in the browser.

### How the screener works

1. **Job description ingestion** – The text area in the Streamlit app captures the target job description. Under the hood, this is tokenised and converted into a TF-IDF vector so it can be compared against resumes.
2. **Resume extraction** – Uploaded PDF resumes are processed with `pdfminer.six` (via `resume_screener/pdf_extractor.py`) to extract raw text. Each resume is stored with an identifier for later ranking.
3. **Preprocessing** – The resume and job description texts are cleaned and tokenised (`resume_screener/preprocessing.py`). This step prepares the text for feature extraction and captures top keywords for highlighting.
4. **Similarity scoring** – `resume_screener/matching.py` builds TF-IDF vectors for every document and computes cosine similarity scores between the job description and each resume. It also surfaces the most influential keywords for transparency.
5. **Results presentation** – `streamlit_app.py` orchestrates the workflow (through `resume_screener/pipeline.py`), displays ranked resumes, and shows similarity scores and keyword highlights directly in the browser.

### Running locally

1. Place your resume PDFs in an accessible folder on your machine.
2. Launch the app (`streamlit run streamlit_app.py`) and open the provided local URL in your browser.
3. Paste the job description text into the input area.
4. Upload one or more resume PDFs via the file uploader.
5. Click **Run screener** to generate similarity scores, review keyword highlights, and download the results as a CSV if needed.
