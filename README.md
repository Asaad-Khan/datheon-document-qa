# Document Q&A Assistant

A retrieval-augmented generation (RAG) application that answers questions about an uploaded PDF and provides source filename and page citations.

Built as a project-based internship task for Datheon.

## Features

- Upload and index one text-based PDF at a time.
- Extract text page by page and normalize whitespace.
- Split text recursively into overlapping passages.
- Generate embeddings locally on CPU.
- Retrieve relevant passages using ChromaDB.
- Supplement semantic retrieval with exact percentage matching.
- Generate document-grounded answers using Groq.
- Display source filename and PDF page citations.
- Return an insufficient-information response when the retrieved evidence cannot answer the question.

## How it works

1. Extract text from each PDF page using pypdf.
2. Normalize whitespace and split each page into chunks of up to 600 characters, with a target overlap of 100 characters.
3. Check that chunks fit the embedding model's token limit.
4. Embed chunks using `sentence-transformers/all-MiniLM-L6-v2`.
5. Store text, embeddings, filenames, and page numbers in Chroma.
6. Retrieve up to five similar passages for a question and supplement them with exact percentage matches.
7. Send the question and retrieved passages to `openai/gpt-oss-20b` through Groq.
8. Validate returned source IDs and construct citations from stored metadata.

The Streamlit upload workflow uses in-memory collections. Replacing the uploaded PDF switches the active collection. Uploaded documents must be indexed again after the app restarts.

## Technology

Python, Streamlit, pypdf, LangChain RecursiveCharacterTextSplitter, Sentence Transformers, ChromaDB, and Groq.

## Local setup

Developed with Python 3.13 on Windows.

Clone this repository and open a terminal in its root directory.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Groq API key.

```dotenv
GROQ_API_KEY=your_groq_api_key_here
```

Launch the application.

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

The first run downloads the embedding model. Internet access is required for that download and for Groq requests.

Upload a text-based PDF, click **Index PDF**, and submit a question.

## Configuration and privacy

Embeddings are generated locally. Answer generation sends the question and retrieved text excerpts to Groq.

Keep API keys in `.env` locally or deployment secrets when hosted. Never commit real API keys or private evaluation documents.

## Evaluation

A development evaluation runner records retrieved passages, raw model responses, answers, citations, latency, and errors.

The included cases cover factual questions, multiple supporting passages, unavailable numerical metrics, submission requirements spread across pages, and a CV question asked against a different document.

To reproduce those checks, place your own copies of these test documents in `data/private`:

- `sample_redacted.pdf`
- `Spark AI-technical skills assessment.pdf`

Then run:

```powershell
.\.venv\Scripts\python.exe evaluate_rag.py --repeats 2
```

Results are saved under `data/private/evaluation`.

Automatic checks assess response status and expected citation pages. Factual correctness, completeness, and citation support require manual review.

Testing identified excessive PDF whitespace as a retrieval problem. Normalization improved passage quality. Repeated runs also showed variation in answer detail, leading to a lower sampling temperature.

These are development tests on two documents, not a benchmark establishing performance on unseen PDFs. The test PDFs are not distributed with this repository.

## Limitations

- Supports selectable-text PDFs. OCR for scanned documents is not implemented.
- Complex layouts, tables, and diagrams may extract poorly.
- Retrieval can miss evidence, particularly for broad questions.
- Answers may omit details, vary across runs, or introduce unsupported claims.
- Valid source IDs establish where citations point, but do not guarantee support for every claim.
- Exact percentage matching is a limited retrieval supplement.
- Questions are independent, without conversational memory.
- The interface handles one PDF at a time.
- Groq availability and account rate limits affect answer generation.

This is an internship prototype. Users should verify important answers against the source document.

## Project files

| File | Purpose |
|---|---|
| `streamlit_app.py` | Upload interface and application state |
| `pdf_index.py` | Shared PDF extraction, chunking, and indexing |
| `ask_pdf.py` | Retrieval, answer generation, and citation validation |
| `evaluate_rag.py` | Development evaluation runner |
| `evaluation_cases.json` | Questions and review criteria |
| `requirements.txt` | Direct dependency versions |
| `.env.example` | API key configuration template |