from io import BytesIO
from uuid import uuid4

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def index_pdf(pdf_bytes, filename, model):
    reader = PdfReader(BytesIO(pdf_bytes))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
    )
    texts = []
    metadata = []

    for page_number, page in enumerate(reader.pages, start=1):
        # Normalize PDF extraction whitespace before chunking.
        page_text = " ".join((page.extract_text() or "").split())
        if not page_text:
            continue

        for chunk in splitter.split_text(page_text):
            texts.append(chunk)
            metadata.append({
                "source": filename,
                "page": page_number,
            })

    if not texts:
        raise ValueError(
            "No selectable text was found. Try a text-based PDF."
        )

    for text in texts:
        token_count = len(
            model.tokenizer(text, truncation=False)["input_ids"]
        )
        if token_count > model.max_seq_length:
            raise ValueError(
                "A passage exceeds the embedding model's token limit."
            )

    embeddings = model.encode(texts, normalize_embeddings=True)
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name=f"upload_{uuid4().hex}",
        embedding_function=None,
    )

    try:
        collection.add(
            ids=[str(number) for number in range(len(texts))],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadata,
        )
    except Exception:
        client.delete_collection(collection.name)
        raise

    return client, collection
