from typing import List, Tuple
import os
import sys
import json
import argparse
import csv
import shutil

# Optional imports for PDF/DOCX extraction
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import docx  # python-docx
except Exception:
    docx = None

from sentence_transformers import SentenceTransformer


def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        if PyPDF2 is None:
            raise RuntimeError("PyPDF2 not installed. pip install PyPDF2")
        text_parts = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                text_parts.append(p.extract_text() or "")
        return "\n".join(text_parts)
    elif ext in (".docx",):
        if docx is None:
            raise RuntimeError("python-docx not installed. pip install python-docx")
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext == ".csv":
        # Read CSV and join rows into lines. Each CSV row becomes one line.
        rows = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for r in reader:
                rows.append(" | ".join(r))
        return "\n".join(rows)
    else:
        # Assume plain text
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[Tuple[int, int, str]]:
    """
    Split text into chunks by characters. Returns list of (start_idx, end_idx, substring).
    Overlap is number of characters to overlap between chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: List[Tuple[int, int, str]] = []
    i = 0
    L = len(text)
    while i < L:
        end = min(i + chunk_size, L)
        chunk = text[i:end]
        chunks.append((i, end, chunk))
        if end == L:
            break
        i = end - overlap
    return chunks


def batch(iterable, n=100):
    """Yield successive n-sized chunks from iterable."""
    it = list(iterable)
    for i in range(0, len(it), n):
        yield it[i : i + n]


def ensure_local_model(model_name: str, download_dir: str) -> str:
    """
    If model_name is a local path, return it.
    If download_dir exists and contains the model, return that path.
    Otherwise download model from HF and save to download_dir/model_safe_name and return that path.
    """
    # If user provided a local path, use it directly
    if os.path.isdir(model_name):
        return model_name

    # If download_dir is not provided, default to ./models/<model_name_safe>
    model_safe = model_name.replace("/", "_").replace(":", "_")
    target_dir = os.path.abspath(os.path.join(download_dir or "models", model_safe))

    if os.path.isdir(target_dir):
        print(f"Using existing local model at {target_dir}")
        return target_dir

    # Try to download and save model to target_dir
    print(f"Downloading model '{model_name}' and saving to {target_dir} ...")
    try:
        m = SentenceTransformer(model_name)
        os.makedirs(target_dir, exist_ok=True)
        m.save(target_dir)
        print(f"Saved model to {target_dir}")
        return target_dir
    except Exception as e:
        # Cleanup partial download if exists
        if os.path.isdir(target_dir):
            try:
                shutil.rmtree(target_dir)
            except Exception:
                pass
        raise RuntimeError(f"Failed to download/save model '{model_name}': {e}") from e


def build_embeddings(model_name: str, chunks: List[Tuple[int, int, str]], batch_size: int = 64):
    model = SentenceTransformer(model_name)
    inputs = [c[2] for c in chunks]
    # The model.encode can handle batching internally; still use slicing to control memory if needed.
    embeddings = []
    for b in batch(inputs, batch_size):
        embs = model.encode(b, batch_size=len(b), show_progress_bar=False, convert_to_numpy=True)
        # ensure list of lists (python floats)
        for e in embs:
            embeddings.append(e.tolist())
    if len(embeddings) != len(chunks):
        raise RuntimeError("Mismatch between chunks and returned embeddings")
    return embeddings


def save_jsonl(out_path: str, file_name: str, chunks: List[Tuple[int, int, str]], embeddings: List[List[float]]):
    with open(out_path, "w", encoding="utf-8") as w:
        for i, ((start, end, text), emb) in enumerate(zip(chunks, embeddings)):
            item = {
                "id": f"{file_name}_chunk_{i}",
                "start": start,
                "end": end,
                "text": text,
                "embedding": emb,
            }
            w.write(json.dumps(item) + "\n")


def main():
    p = argparse.ArgumentParser(prog="build_embeddings.py", description="Build embeddings for a file using HuggingFace SentenceTransformers.")
    p.add_argument("path", help="Path to the input file (txt, md, pdf, docx, csv supported)")
    p.add_argument("-m", "--model", default="all-MiniLM-L6-v2", help="SentenceTransformers model to use (default: all-MiniLM-L6-v2). Accepts HF model name or local path.")
    p.add_argument("-o", "--out", default=None, help="Output JSONL path (default: <input>.emb.jsonl)")
    p.add_argument("--chunk-size", type=int, default=2000, help="Chunk size in characters")
    p.add_argument("--overlap", type=int, default=200, help="Overlap size in characters")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding encoder")
    p.add_argument("--download-model", action="store_true", help="If set, download the model locally before use (saved to --download-dir or ./models).")
    p.add_argument("--download-dir", default=None, help="Directory to save downloaded model when --download-model is set (default: ./models).")
    args = p.parse_args()

    text = extract_text(args.path)
    chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"Created {len(chunks)} chunks from {args.path}")

    model_source = args.model
    if args.download_model:
        model_source = ensure_local_model(args.model, args.download_dir)

    embeddings = build_embeddings(model_source, chunks, batch_size=args.batch_size)
    out_path = args.out or (args.path + ".emb.jsonl")
    base_name = os.path.splitext(os.path.basename(args.path))[0]
    save_jsonl(out_path, base_name, chunks, embeddings)
    print(f"Saved embeddings to {out_path}")


if __name__ == "__main__":
    main()