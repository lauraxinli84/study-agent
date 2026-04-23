"""
Minimal in-memory vector store backed by OpenAI embeddings.

Chose this over Pinecone/Weaviate/pgvector because:
  - zero external service required (works on HF Spaces free tier)
  - course materials fit easily in memory (< a few thousand chunks)
  - scikit-learn cosine_similarity is accurate and trivial
"""
from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
VECTORSTORE_PATH = Path("data/vectorstore/store.pkl")


@dataclass
class Chunk:
    doc_name: str
    chunk_idx: int
    text: str


@dataclass
class VectorStore:
    chunks: list[Chunk] = field(default_factory=list)
    embeddings: np.ndarray | None = None

    def is_empty(self) -> bool:
        return not self.chunks

    def doc_names(self) -> list[str]:
        return sorted({c.doc_name for c in self.chunks})


# ---- Chunking ---------------------------------------------------------------

def _chunk_text(text: str, target_words: int = 250, overlap: int = 40) -> list[str]:
    """Split on paragraph boundaries, then pack into ~target_words chunks."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_wc = 0
    for p in paragraphs:
        wc = len(p.split())
        if buf_wc + wc > target_words and buf:
            chunks.append("\n\n".join(buf))
            # Carry a small overlap (last few words) for context continuity
            tail_words = " ".join(buf).split()[-overlap:]
            buf = [" ".join(tail_words)] if tail_words else []
            buf_wc = len(tail_words)
        buf.append(p)
        buf_wc += wc
    if buf:
        chunks.append("\n\n".join(buf))
    return [c for c in chunks if c.strip()]


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if path.suffix.lower() in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {path.suffix}")


# ---- Embedding --------------------------------------------------------------

def _embed_batch(client: OpenAI, texts: list[str]) -> np.ndarray:
    # OpenAI accepts up to 2048 inputs per call; batch conservatively.
    out: list[list[float]] = []
    BATCH = 64
    for i in range(0, len(texts), BATCH):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i : i + BATCH])
        out.extend(d.embedding for d in resp.data)
    return np.array(out, dtype=np.float32)


# ---- Public API -------------------------------------------------------------

def load_store() -> VectorStore:
    if VECTORSTORE_PATH.exists():
        with open(VECTORSTORE_PATH, "rb") as f:
            return pickle.load(f)
    return VectorStore()


def save_store(store: VectorStore) -> None:
    VECTORSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VECTORSTORE_PATH, "wb") as f:
        pickle.dump(store, f)


def add_document(store: VectorStore, file_path: Path, client: OpenAI) -> int:
    """Read, chunk, embed, and append a document to the store. Returns #chunks added."""
    text = _read_file(file_path)
    pieces = _chunk_text(text)
    if not pieces:
        return 0
    new_chunks = [
        Chunk(doc_name=file_path.name, chunk_idx=i, text=t) for i, t in enumerate(pieces)
    ]
    new_embeds = _embed_batch(client, pieces)
    store.chunks.extend(new_chunks)
    if store.embeddings is None:
        store.embeddings = new_embeds
    else:
        store.embeddings = np.vstack([store.embeddings, new_embeds])
    save_store(store)
    return len(new_chunks)


def remove_document(store: VectorStore, doc_name: str) -> int:
    """Remove all chunks of a given document. Returns #chunks removed."""
    if store.is_empty():
        return 0
    keep = [i for i, c in enumerate(store.chunks) if c.doc_name != doc_name]
    removed = len(store.chunks) - len(keep)
    store.chunks = [store.chunks[i] for i in keep]
    store.embeddings = store.embeddings[keep] if store.embeddings is not None else None
    if not store.chunks:
        store.embeddings = None
    save_store(store)
    return removed


def search(store: VectorStore, query: str, client: OpenAI, k: int = 4) -> list[dict]:
    if store.is_empty() or store.embeddings is None:
        return []
    q_emb = _embed_batch(client, [query])
    sims = cosine_similarity(q_emb, store.embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:k]
    results = []
    for i in top_idx:
        c = store.chunks[int(i)]
        results.append(
            {
                "doc_name": c.doc_name,
                "chunk_idx": c.chunk_idx,
                "text": c.text,
                "score": float(sims[int(i)]),
            }
        )
    return results
