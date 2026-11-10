from hashlib import sha256

import streamlit as st
from sentence_transformers import SentenceTransformer

from ask_pdf import answer_question
from pdf_index import index_pdf


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )




st.set_page_config(page_title="Document Q&A")
st.title("Document Q&A")
st.caption(
    "Upload a text-based PDF and ask questions about it. "
    "When you ask a question, Groq receives the question and retrieved excerpts."
)

uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
pdf_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
upload_key = (
    (uploaded_file.name, sha256(pdf_bytes).hexdigest())
    if pdf_bytes is not None
    else None
)

active = st.session_state.get("active_pdf")

if active is not None and active["key"] != upload_key:
    active["client"].delete_collection(active["collection"].name)
    del st.session_state["active_pdf"]
    active = None

if uploaded_file is None:
    st.info("Upload a PDF to get started.")
    st.stop()

if active is None:
    if not st.button("Index PDF"):
        st.stop()

    try:
        with st.spinner("Reading and indexing the PDF..."):
            model = load_embedding_model()
            client, collection = index_pdf(
                pdf_bytes, uploaded_file.name, model
            )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except Exception:
        st.error("Could not index this PDF. Check that it opens without a password.")
        st.stop()

    active = {"key": upload_key, "client": client, "collection": collection}
    st.session_state["active_pdf"] = active

collection = active["collection"]
st.success(f"Ready: {uploaded_file.name} ({collection.count()} passages)")

with st.form("question_form"):
    question = st.text_input("Your question")
    submitted = st.form_submit_button("Ask")

if submitted:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            with st.spinner("Searching the PDF and preparing an answer..."):
                answer = answer_question(
                    question,
                    collection=collection,
                    model=load_embedding_model(),
                )
        except Exception as exc:
            st.error(
                f"Could not answer the question ({type(exc).__name__}). "
                "Check that Groq is configured and try again."
            )
        else:
            st.subheader("Answer")
            st.text(answer)
