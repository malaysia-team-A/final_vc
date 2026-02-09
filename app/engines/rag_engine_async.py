import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from app.engines.rag_engine import rag_engine as _sync_rag_engine

class AsyncRAGEngine:
    def __init__(self):
        # Create a thread pool for CPU-bound tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        print("[AsyncRAG] Initialized ThreadPool for FAISS operations")

    async def search_context(
        self,
        query: str,
        top_k: int = 5,
        preferred_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run blocking FAISS search in a separate thread.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: _sync_rag_engine.search(
                query,
                n_results=top_k,
                preferred_labels=preferred_labels,
            ),
        )

    async def index_mongodb_collections(self) -> int:
        """
        Run blocking indexing in a separate thread.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            _sync_rag_engine.index_mongodb_collections
        )

    async def add_document(self, text: str, source: str) -> bool:
        """
        Backward-compatible helper for old callers that push raw text.
        Writes a temporary text file, then ingests through the sync engine.
        """
        import os
        import tempfile

        filename = os.path.basename(str(source or "uploaded.txt"))
        suffix = os.path.splitext(filename)[1].lower() or ".txt"
        if suffix not in {".txt", ".csv"}:
            suffix = ".txt"

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix, delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(str(text or ""))
                tmp_path = tmp.name
            return await self.ingest_file(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    async def ingest_file(self, file_path: str) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            _sync_rag_engine.ingest_file,
            file_path,
        )

# Singleton
rag_engine_async = AsyncRAGEngine()
