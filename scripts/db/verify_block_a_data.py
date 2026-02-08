
import sys
import os
import time
from pymongo import MongoClient
from dotenv import load_dotenv

sys.path.append(os.getcwd())
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

print("Starting verification...", flush=True)

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    db = client["UCSI_DB"]
    print("Connected to DB.", flush=True)
    
    # Check if 'Block A' exists in Facility or Hostel via Regex
    print("Searching for 'Block A' via Regex...", flush=True)
    
    facs = list(db["UCSI_FACILITY"].find({"name": {"$regex": "Block A", "$options": "i"}}).limit(2))
    print(f"Facilities found: {len(facs)}", flush=True)
    for f in facs: print(f" - {f.get('name')}", flush=True)
    
    hostels = list(db["Hostel"].find({"building": {"$regex": "Block A", "$options": "i"}}).limit(2))
    print(f"Hostels found: {len(hostels)}", flush=True)
    for h in hostels: print(f" - {h.get('building')}", flush=True)
    
except Exception as e:
    print(f"Error: {e}", flush=True)
