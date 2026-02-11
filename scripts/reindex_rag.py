
import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from app.engines.db_engine_async import db_engine_async
from app.engines.rag_engine_async import rag_engine_async
from app.config import Config

async def main():
    print(f"Connecting to MongoDB: {Config.MONGO_URI}...")
    await db_engine_async.connect()
    print("MongoDB Connected.")
    
    print("Rebuilding RAG Index (including new 'Student' collection)...")
    try:
        count = await rag_engine_async.index_mongodb_collections()
        print(f"\n[SUCCESS] Rebuilt index with {count} documents.")
        
        # Verify index info
        info = await rag_engine_async.get_index_info()
        print(f"Index Info: {info}")
        
    except Exception as e:
        print(f"[ERROR] Failed to rebuild index: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
