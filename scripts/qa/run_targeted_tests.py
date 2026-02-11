import asyncio
import aiohttp
import json
import time
import statistics
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.config import Config

BASE_URL = "http://localhost:5000/api/chat"
headers = {"Content-Type": "application/json"}

async def send_query(session, query, conversation_id=None):
    payload = {
        "message": query,
        "conversation_id": conversation_id,
        "user_id": "test_qa_user"
    }
    start_time = time.time()
    try:
        async with session.post(BASE_URL, json=payload, headers=headers) as resp:
            data = await resp.json()
            latency = time.time() - start_time
            return {
                "query": query,
                "status": resp.status,
                "latency": latency,
                "response": data,
                "success": resp.status == 200
            }
    except Exception as e:
        return {
            "query": query,
            "status": 0,
            "latency": 0,
            "response": str(e),
            "success": False
        }

async def run_batch(queries, batch_name, concurrency=5):
    print(f"\n--- Running {batch_name} (Total: {len(queries)}, Concurrency: {concurrency}) ---")
    results = []
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [send_query(session, q) for q in queries]
        # Run with progress indicator is hard in async simple script, so just gather all
        results = await asyncio.gather(*tasks)
    
    return results

def analyze_general_results(results):
    print("\n--- General Knowledge Test Analysis ---")
    evasive_count = 0
    success_count = 0
    
    # Evasive keywords indicating "I don't know" style failure when general knowledge should be used
    evasive_keywords = [
        "I do not have information", 
        "no information provided", 
        "cannot answer",
        "context does not contain",
        "I'm sorry",
        "I apologize"
    ]
    
    for r in results:
        if not r["success"]:
            continue
            
        success_count += 1
        text = r["response"].get("response", "") or r["response"].get("text", "")
        # Check if response is evasive
        is_evasive = any(kw.lower() in text.lower() for kw in evasive_keywords)
        # However, for general queries, we EXPECT some level of "check the website" 
        # but we want to fail if it strictly says "context missing, cannot answer" WITHOUT giving general info.
        # This is a heuristic check.
        
        # A good general answer usually contains "UCSI" and some facts, 
        # while a bad one is short and says "I don't know".
        if len(text) < 50 and is_evasive: 
            evasive_count += 1
            # print(f"[Potential Fail] {r['query']} -> {text}")

    print(f"Total: {len(results)}")
    print(f"Server Errors: {len(results) - success_count}")
    print(f"Potential Evasive/Short Responses: {evasive_count}")
    print(f"Success Rate (HTTP 200): {success_count/len(results)*100:.1f}%")

def analyze_rag_results(results):
    print("\n--- RAG Specific Test Analysis ---")
    link_mismatch_count = 0
    success_count = 0
    
    for r in results:
        if not r["success"]:
            continue
            
        success_count += 1
        resp_data = r["response"]
        text = resp_data.get("response", "") or resp_data.get("text", "")
        rich = resp_data.get("rich_content", {})
        links = rich.get("links", [])
        
        # Check for link correctness logic:
        # If links exist, the name in the label MUST appear in the text.
        for lnk in links:
            if lnk.get("type") == "staff_profile":
                label = lnk.get("label", "").replace("View ", "").replace("'s Profile", "").lower()
                if label and label not in text.lower():
                    link_mismatch_count += 1
                    print(f"[Link Mismatch] Query: {r['query']}")
                    print(f"  Link Label: {label}")
                    print(f"  Response Text: {text[:100]}...")

    print(f"Total: {len(results)}")
    print(f"Server Errors: {len(results) - success_count}")
    print(f"Suspicious Link Discrepancies: {link_mismatch_count}")
    print(f"Success Rate (HTTP 200): {success_count/len(results)*100:.1f}%")

async def main():
    # 1. Generate data
    print("Generating test data...")
    os.system("python scripts/qa/generate_test_data.py")
    
    with open("test_cases/generated/general_queries.json", "r", encoding="utf-8") as f:
        general_queries = json.load(f)
        
    with open("test_cases/generated/rag_queries.json", "r", encoding="utf-8") as f:
        rag_queries = json.load(f)

    # 2. Run General Test (Sequential-ish to avoid being blocked by rate limit too fast, but we want stress too)
    # Let's do partial concurrency
    t0 = time.time()
    general_results = await run_batch(general_queries, "General Queries", concurrency=10)
    analyze_general_results(general_results)
    
    # 3. Run RAG Test
    rag_results = await run_batch(rag_queries, "RAG Queries", concurrency=10)
    analyze_rag_results(rag_results)
    
    # 4. Stress Test (Mixed)
    print("\n--- Running Combined Stress Test (200 mixed queries, high concurrency) ---")
    mixed_queries = (general_queries + rag_queries)[:200]
    stress_results = await run_batch(mixed_queries, "Stress Test", concurrency=20)
    
    latencies = [r["latency"] for r in stress_results if r["success"]]
    if latencies:
        print(f"Avg Latency: {statistics.mean(latencies):.2f}s")
        print(f"Max Latency: {max(latencies):.2f}s")
        print(f"P95 Latency: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}s")
    
    total_time = time.time() - t0
    print(f"\nTotal Test Duration: {total_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
