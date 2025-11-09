# retrievers.py
import os
import math
from typing import List, Any
import pandas as pd
import re

# Try to import llama_index style modules; if unavailable, we'll fall back.
try:
    from llama_index import (
        GPTVectorStoreIndex,
        SimpleDirectoryReader,
        LLMPredictor,
        ServiceContext,
        download_loader,
    )
    from llama_index.node_parser import SimpleNodeParser
    LLAMA_INDEX_AVAILABLE = True
except Exception:
    LLAMA_INDEX_AVAILABLE = False


# ----------------------
# Lightweight in-memory retriever fallback
# ----------------------
class SimpleDoc:
    def __init__(self, text: str, meta: dict = None):
        self.text = text
        self.meta = meta or {}

class SimpleRetriever:
    def __init__(self, docs: List[SimpleDoc]):
        self.docs = docs

    def retrieve(self, query: str, k: int = 1) -> List[SimpleDoc]:
        """
        Return top-k docs by simple token overlap scoring.
        Not an embedding-based retriever; used as a safe fallback.
        """
        if not query:
            return self.docs[:k]
        q_tokens = set(re.findall(r"\w+", query.lower()))
        def score(doc: SimpleDoc):
            d_tokens = set(re.findall(r"\w+", doc.text.lower()))
            overlap = len(q_tokens & d_tokens)
            # also boost by doc length similarity to query length
            length_score = 1.0 / (1 + abs(len(d_tokens) - len(q_tokens)))
            return overlap + length_score * 0.01
        sorted_docs = sorted(self.docs, key=score, reverse=True)
        return sorted_docs[:k]

    # `as_retriever` compatibility with llama_index-like callsites.
    def as_retriever(self, k: int = 1, similarity_top_k: int = 1):
        # Return a thin wrapper exposing .retrieve(query)[0].text
        parent = self
        class _R:
            def retrieve(self_inner, query: str):
                return parent.retrieve(query, k=k)
        return _R()


# ----------------------
# Factory functions
# ----------------------
def build_dataframe_retriever_from_df(df: pd.DataFrame, max_preview_chars: int = 800):
    """
    Build a retriever that turns the dataframe into small textual documents.
    Each document summarizes some rows/columns.
    Uses llama_index if available, otherwise returns SimpleRetriever.
    """
    # Basic textualization: columns + first N rows, then row summaries
    cols = ", ".join(list(df.columns[:10]))
    header = f"DataFrame with {len(df)} rows and columns: {cols}."
    # create row-summary docs (cap at 500 rows)
    docs = []
    docs.append(SimpleDoc(header, meta={"type": "meta"}))
    n_rows = min(500, len(df))
    for i in range(n_rows):
        row = df.iloc[i]
        row_text = " | ".join([f"{c}:{str(row[c])}" for c in df.columns[:10]])
        docs.append(SimpleDoc(f"row_{i}: {row_text}"))
    return SimpleRetriever(docs)


def build_style_retriever_from_texts(style_texts: List[str]):
    """
    Build a retriever over style guidance text blocks (css, instructions, examples).
    """
    docs = [SimpleDoc(t, meta={"type": "style", "idx": i}) for i, t in enumerate(style_texts)]
    return SimpleRetriever(docs)


# ----------------------
# Convenience loader: accept CSV path or dataframe directly
# ----------------------
def create_retrievers_from_csv(csv_path: str = None, df: pd.DataFrame = None, style_texts: List[str] = None):
    """
    Returns a dict with:
    - 'dataframe_index': retriever-like object
    - 'style_index': retriever-like object
    """
    if df is None and csv_path:
        df = pd.read_csv(csv_path)
    if df is None:
        raise ValueError("Either df or csv_path must be provided.")

    style_texts = style_texts or ["Default styling: prefer Plotly charts and concise captions."]

    dataframe_retriever = build_dataframe_retriever_from_df(df)
    style_retriever = build_style_retriever_from_texts(style_texts)

    return {"dataframe_index": dataframe_retriever, "style_index": style_retriever}
