from typing import List, Dict, Any, Optional
import os
import chromadb


class VectorIndex:
    """
    Chroma-based simple vector index per session. Stores chunk summaries with metadata.
    """

    def __init__(self, session_id: str, persistent_dir: str, embedding_model: str):
        self.session_id = session_id
        self.embedding_model = embedding_model
        os.makedirs(persistent_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=os.path.join(persistent_dir, "chroma"))
        self.collection = self.client.get_or_create_collection(name=f"session_{session_id}")

    def upsert(self, items: List[Dict[str, Any]]):
        ids = [it["id"] for it in items]
        docs = [f"{it.get('summary','')}\ncols:{','.join(it.get('columns', []))}" for it in items]
        metadatas = [
            {
                "row_range": it.get("row_range"),
                "columns": it.get("columns"),
                "summary": it.get("summary"),
                "examples": it.get("examples", []),
            }
            for it in items
        ]
        self.collection.upsert(ids=ids, documents=docs, metadatas=metadatas)

    def search(self, query: str, top_k: int = 8, offset: int = 0) -> List[Dict[str, Any]]:
        try:
            res = self.collection.query(query_texts=[query], n_results=max(1, top_k + offset))
        except Exception:
            return []

        out: List[Dict[str, Any]] = []
        ids_all = (res.get("ids") or [[]])[0]
        metas_all = (res.get("metadatas") or [[]])[0]
        dists_all = None
        if res.get("distances") is not None:
            dists_all = (res.get("distances") or [[]])[0]
        elif res.get("embeddings") is not None:
            dists_all = (res.get("embeddings") or [[]])[0]

        if not ids_all:
            return []

        end = offset + top_k
        sliced_ids = ids_all[offset:end] if len(ids_all) > offset else []
        for i, doc_id in enumerate(sliced_ids):
            meta_idx = offset + i
            m = metas_all[meta_idx] if meta_idx < len(metas_all) and metas_all else {}
            score = dists_all[meta_idx] if dists_all and meta_idx < len(dists_all) else 0.0
            out.append({
                "id": doc_id,
                "summary": (m or {}).get("summary"),
                "row_range": (m or {}).get("row_range"),
                "examples": (m or {}).get("examples", []),
                "score": float(score) if isinstance(score, (int, float)) else 0.0,
                "has_more": len(ids_all) > end,
                "next_offset": end,
            })
        return out



