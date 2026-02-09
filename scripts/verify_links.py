import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from app.engines.rag_engine import rag_engine
from app.engines.db_engine import db_engine

async def verify():
    print("Connecting to DB...")
    db_engine.connect()
    
    print("Re-indexing collections...")
    count = rag_engine.index_mongodb_collections()
    print(f"Indexed {count} documents.")
    
    print("\n[Test 1] Searching for 'Foundation in Arts' (Major)...")
    result = rag_engine.search("Foundation in Arts")
    context = result.get("context", "")
    if "Url:" in context:
        print("✅ SUCCESS: URL found in Major context.")
    else:
        print("❌ FAIL: URL NOT found in Major context.")
        # print("Context head:", context[:200])
        
    print("\n[Test 2] Searching for Staff members...")
    result = rag_engine.search("lecturer")
    context = result.get("context", "")
    if "profile_url:" in context:
        print("✅ SUCCESS: Profile URL found in Staff context.")
    else:
        print("❌ FAIL: Profile URL NOT found in Staff context.")
    
    db_engine.close()

if __name__ == "__main__":
    asyncio.run(verify())
