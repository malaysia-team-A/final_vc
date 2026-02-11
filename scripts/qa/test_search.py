
import asyncio
from app.engines.rag_engine_async import rag_engine_async
from app.engines.db_engine_async import db_engine_async

async def main():
    await db_engine_async.connect()
    # Build index first (important!)
    count = await rag_engine_async.index_mongodb_collections()
    print(f"Indexed {count} documents.")
    
    queries = [
        "Vicky Yiran",
        "체크인 시간은 언제인가요?",
        "애완동물 키울 수 있나요?",
        "Can I bring my cat?",
        "Block A 위치",
        "성적 확인",
        "How to check my GPA?",
    ]

    for q in queries:
        print(f"\nQUERY: {q}")
        try:
            results = await rag_engine_async.search_context(q, top_k=3)
            print(f"RAG FOUND: {len(results or [])} items")
            for res in (results or []):
                 print(f" - {res.text[:100]}... (Source: {res.metadata.get('source')})")
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(main())
