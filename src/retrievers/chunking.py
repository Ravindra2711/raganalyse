from typing import List, Dict, Any, Iterable


class RowRangeChunker:
    """
    Simple row-range chunker stub. In production, preserve group continuity and add per-chunk stats.
    """

    def __init__(self, chunk_size: int = 400):
        self.chunk_size = max(50, chunk_size)

    def chunk(self, df_like: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = list(df_like)
        chunks = []
        for i in range(0, len(rows), self.chunk_size):
            part = rows[i : i + self.chunk_size]
            chunk_id = f"{i}-{i+len(part)-1}"
            summary = f"Rows {i}-{i+len(part)-1}; {len(part)} rows"
            examples = part[0: min(5, len(part))]
            chunks.append({
                "id": chunk_id,
                "row_range": [i, i + len(part) - 1],
                "summary": summary,
                "examples": examples,
            })
        return chunks




