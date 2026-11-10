"""Run existing RAG behavior and save evidence for manual review."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import time

from sentence_transformers import SentenceTransformer

from ask_pdf import answer_question
from pdf_index import index_pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents-dir", type=Path, default=Path("data/private"))
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    root = Path(__file__).resolve().parent
    cases = json.loads((root / "evaluation_cases.json").read_text())
    documents = args.documents_dir.resolve()
    for case in cases:
        if not (documents / case["document"]).is_file():
            parser.error(f"Missing PDF: {documents / case['document']}")

    output = root / "data" / "private" / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result_path = output / f"{run_id}.jsonl"
    manifest = {
        "run_id": run_id,
        "repeats": args.repeats,
        "code_sha256": {
            name: sha256((root / name).read_bytes()).hexdigest()
            for name in ["ask_pdf.py", "pdf_index.py", "evaluation_cases.json"]
        },
        "document_sha256": {
            name: sha256((documents / name).read_bytes()).hexdigest()
            for name in dict.fromkeys(case["document"] for case in cases)
        },
        "packages": {
            name: version(name)
            for name in ["chromadb", "groq", "sentence-transformers",
                         "pypdf", "langchain-text-splitters"]
        },
        "grading": "Status and page coverage checks only. Semantic correctness, completeness, and citation support require manual review.",
    }
    result_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    # Reindex each document with the same code used by Streamlit.
    with result_path.open("x", encoding="utf-8") as log:
        for filename in dict.fromkeys(case["document"] for case in cases):
            client, collection = index_pdf(
                (documents / filename).read_bytes(), filename, model
            )
            try:
                for case in cases:
                    if case["document"] != filename:
                        continue
                    for repeat in range(1, args.repeats + 1):
                        trace = {}
                        record = {"case": case, "repeat": repeat}
                        started = time.perf_counter()
                        try:
                            record["answer"] = answer_question(
                                case["question"], collection=collection,
                                model=model, trace=trace,
                            )
                        except Exception as exc:
                            # Keep errors distinct from evidence-based refusals.
                            trace["failure_stage_status"] = trace.get("status")
                            trace["status"] = "error"
                            record["error_type"] = type(exc).__name__
                        record["seconds"] = round(time.perf_counter() - started, 3)
                        record["trace"] = trace
                        cited_pages = {
                            trace["sources"][sid]["page"]
                            for sid in trace.get("cited_source_ids", [])
                        }
                        record["expected_behavior_matches"] = (
                            trace.get("status") == (
                                "abstained" if case["expect_abstention"] else "answered"
                            )
                        )
                        record["expected_pages_cited"] = (
                            set(case["expected_pages"]).issubset(cited_pages)
                            if not case["expect_abstention"] else None
                        )
                        record["manual_review"] = {
                            "correctness": None, "completeness": None,
                            "citation_support": None, "notes": "",
                        }
                        log.write(json.dumps(record, ensure_ascii=False) + "\n")
                        log.flush()
                        print(f"{case['id']} run {repeat}: {trace.get('status')}", flush=True)
                        time.sleep(3)
            finally:
                client.delete_collection(collection.name)
    print(f"Saved results: {result_path}")
    print("Review the answers and passages against each case's criteria.")


if __name__ == "__main__":
    main()
