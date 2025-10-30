from typing import Dict, Any, List, Optional, Tuple
import time

from src.utils.logger import Logger
from src.retrievers.chunking import RowRangeChunker
from src.embeddings.index import VectorIndex

logger = Logger(name="autogen_orchestrator", see_time=True, console_log=False)


class ContextPackage:
    def __init__(self, header: str, chunks: List[Dict[str, Any]], provenance: List[Dict[str, Any]],
                 next_offset: Optional[int] = None):
        self.header = header
        self.chunks = chunks
        self.provenance = provenance
        self.next_offset = next_offset


class Orchestrator:
    """
    Minimal orchestration skeleton. For now, this uses retrieval to build a context header
    and calls a provider via src.utils.model_registry (OpenAI/LiteLLM) to produce an answer.
    Later this can be upgraded to full Autogen GroupChat.
    """

    def __init__(self, persistent_dir: str = ".vector_indexes"):
        self.persistent_dir = persistent_dir

    def build_context(
        self,
        session_id: str,
        df_info: Dict[str, Any],
        query: str,
        embedding_model: str = "text-embedding-3-small",
        k: int = 8,
        token_budget: int = 12000,
        offset: int = 0,
    ) -> ContextPackage:
        index = VectorIndex(session_id=session_id, persistent_dir=self.persistent_dir, embedding_model=embedding_model)

        results = index.search(query=query, top_k=k, offset=offset)

        header = (
            f"Dataset: {df_info.get('name','dataset')}\n"
            f"Schema: {', '.join(df_info.get('columns', []))}\n"
            "Instructions: Answer concisely. Cite chunk_id and row_range when using data."
        )

        chunks = []
        provenance = []
        total_tokens_est = 0
        for r in results:
            chunk = {
                "chunk_id": r["id"],
                "summary": r.get("summary", ""),
                "row_range": r.get("row_range"),
                "examples": r.get("examples", []),
            }
            chunks.append(chunk)
            provenance.append({
                "chunk_id": r["id"],
                "row_range": r.get("row_range"),
                "score": float(r.get("score", 0.0)),
            })
            total_tokens_est += len(chunk["summary"].split()) + 20

        next_offset = None
        if results and results[-1].get("has_more"):
            next_offset = results[-1]["next_offset"]

        return ContextPackage(header=header, chunks=chunks, provenance=provenance, next_offset=next_offset)

    def answer_query(
        self,
        chat_id: int,
        session_id: str,
        df_info: Dict[str, Any],
        query: str,
        model_name: str,
        temperature: float = 0.1,
        max_tokens: int = 800,
        embedding_model: str = "text-embedding-3-small",
        k: int = 8,
        offset: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        from src.utils.model_registry import completion_call, get_provider_for_model

        started = time.perf_counter()
        context_pkg = self.build_context(
            session_id=session_id,
            df_info=df_info,
            query=query,
            embedding_model=embedding_model,
            k=k,
            offset=offset,
        )

        context_text_parts = [context_pkg.header]
        for c in context_pkg.chunks:
            context_text_parts.append(
                f"\n[chunk_id={c['chunk_id']} rows={c['row_range']}]\n{c['summary']}\n"
            )
        context_text = "\n".join(context_text_parts)

        system_prompt = (
            "You are a data analysis assistant. Use the provided dataset context. "
            "Always cite chunk_id and row_range when referencing data."
        )
        user_prompt = (
            f"Context:\n{context_text}\n\nQuestion: {query}\n"
            "Return a concise answer. If uncertain, say what else is needed."
        )

        provider = get_provider_for_model(model_name)
        resp = completion_call(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        usage = resp.get("usage", {})
        text = resp.get("content") or resp.get("text") or resp.get("message", "")

        meta = {
            "provider": provider,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
            "request_time_ms": elapsed_ms,
            "provenance": context_pkg.provenance,
            "next_offset": context_pkg.next_offset,
        }
        return text, meta




