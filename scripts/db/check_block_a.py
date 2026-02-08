
import os
from pymongo import MongoClient
from dotenv import load_dotenv
import time

load_dotenv()

def log(msg):
    print(msg)

MONGO_URI = os.getenv("MONGO_URI")

try:
    log(f"Connecting to MongoDB...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client["UCSI_DB"]
    
    # Check collections
    log(f"Collections: {db.list_collection_names()}")
    
    # 1. Inspect Data
    log("\n--- Inspecting Data for 'Block A' ---")
    facilities = list(db["UCSI_FACILITY"].find({"$or": [{"name": {"$regex": "Block A", "$options": "i"}}, {"location": {"$regex": "Block A", "$options": "i"}}]}))
    hostels = list(db["Hostel"].find({"$or": [{"building": {"$regex": "Block A", "$options": "i"}}, {"room_type": {"$regex": "Block A", "$options": "i"}}]}))
    
    log(f"Found {len(facilities)} facilities matching 'Block A' (Regex)")
    for f in facilities:
        log(f" - Facility: {f.get('name')}, Loc: {f.get('location')}")
        
    log(f"Found {len(hostels)} hostels matching 'Block A' (Regex)")
    for h in hostels:
        log(f" - Hostel: {h.get('building')}, Type: {h.get('room_type')}")
        
    # 2. Test Text Search (Simulating rag_engine.py)
    log("\n--- Testing Text Search for 'Block A' ---")
    
    # Ensure indexes (copied from rag_engine.py)
    try:
        db['Hostel'].create_index([("room_type", "text"), ("building", "text")])
        db['UCSI_FACILITY'].create_index([("name", "text"), ("location", "text")])
    except Exception as e:
        log(f"Index warning: {e}")
        
    query = "Block A"
    search_query = {"$text": {"$search": query}}
    
    log(f"Running Text Search: {search_query}")
    
    fac_results = list(db["UCSI_FACILITY"].find(search_query))
    hostel_results = list(db["Hostel"].find(search_query))
    
    log(f"Text Search Found {len(fac_results)} facilities")
    for f in fac_results:
         log(f" - {f.get('name')}")

    log(f"Text Search Found {len(hostel_results)} hostels")
    for h in hostel_results:
         log(f" - {h.get('building')}")

    # 3. Test Text Search for 'Block D'
    log("\n--- Testing Text Search for 'Block D' ---")
    query_d = "Block D"
    search_query_d = {"$text": {"$search": query_d}}
    hostel_results_d = list(db["Hostel"].find(search_query_d))
    log(f"Text Search Found {len(hostel_results_d)} hostels for 'Block D'")
    for h in hostel_results_d:
         log(f" - {h.get('building')}")

except Exception as e:
    log(f"Error: {e}")
