
import sys
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Add app to path
sys.path.append(os.getcwd())

load_dotenv()

# Simulation of the Unified Algorithm in rag_engine.py
# (We define it here to test without dealing with complex app imports for now)

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["UCSI_DB"]

def smart_search_collection(coll_name, text_fields, label, query, limit=2):
    coll = db[coll_name]
    found_docs = []
    seen_ids = set()

    print(f"--- Searching {coll_name} for '{query}' ---")

    # A. Text Search
    try:
        cursor = coll.find(
            {"$text": {"$search": query}}, 
            {"score": {"$meta": "textScore"}, "_id": 0}
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)
        
        count_text = 0
        for doc in cursor:
            count_text += 1
            s_doc = str(doc)
            if s_doc not in seen_ids:
                seen_ids.add(s_doc)
                found_docs.append(doc)
        print(f"  [Text Search] Found: {count_text}")
    except Exception as e: 
        print(f"  [Text Search] Error/No Index: {e}")

    # B. Regex Search (Fallback)
    if len(found_docs) < limit or len(query) < 10:
        try:
            regex_conditions = []
            cleaned_q = query.strip()
            for field in text_fields:
                regex_conditions.append({field: {"$regex": cleaned_q, "$options": "i"}})
            
            cursor = coll.find({"$or": regex_conditions}, {"_id": 0}).limit(limit)
            count_regex = 0
            for doc in cursor:
                count_regex += 1
                s_doc = str(doc)
                if s_doc not in seen_ids:
                    seen_ids.add(s_doc)
                    found_docs.append(doc)
            print(f"  [Regex Search] Found: {count_regex}")
        except Exception as e: 
            print(f"  [Regex Search] Error: {e}")

    # Results
    print(f"  => Total Unique Items: {len(found_docs)}")
    for d in found_docs:
        name = d.get('name') or d.get('Programme') or d.get('room_type') or d.get('building')
        print(f"     - Found: {name}")

# Test Cases
queries = ["Block A", "Block D", "Computer Science"]
domains = [
    ('UCSI_FACILITY', ['name', 'location'], 'Facility Info'),
    ('Hostel', ['room_type', 'building', 'location'], 'Hostel Info')
]

print("=== VERIFYING UNIVERSAL ALGORITHM ===")
for q in queries:
    print(f"\nQuery: {q}")
    for coll, fields, label in domains:
        smart_search_collection(coll, fields, label, q)

