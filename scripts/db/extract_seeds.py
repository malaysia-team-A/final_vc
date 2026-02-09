
import os
import json
import random
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def extract_seed_data():
    client = MongoClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=5000)
    db = client["UCSI_DB"]
    
    seeds = {
        "programmes": [],
        "facilities": [],
        "staff": [],
        "hostels": []
    }
    
    # 1. Programmes
    print("Fetching Programmes...")
    progs = db['UCSI_ MAJOR'].find({}, {"Programme": 1, "_id": 0}).limit(200)
    seeds["programmes"] = [p["Programme"] for p in progs if "Programme" in p]
    
    # 2. Facilities
    print("Fetching Facilities...")
    facs = db['UCSI_FACILITY'].find({}, {"name": 1, "_id": 0}).limit(50)
    seeds["facilities"] = [f["name"] for f in facs if "name" in f]
    
    # 3. Staff
    print("Fetching Staff...")
    staff_docs = db['UCSI_STAFF'].find({}, {"staff_members": 1, "_id": 0}).limit(20)
    for doc in staff_docs:
        if "staff_members" in doc:
            for s in doc["staff_members"]:
                if "name" in s:
                    seeds["staff"].append(s["name"])
    
    # 4. Hostels
    print("Fetching Hostels...")
    hostels = db['Hostel'].find({}, {"room_type": 1, "_id": 0}).limit(20)
    seeds["hostels"] = [h["room_type"] for h in hostels if "room_type" in h]
    
    # Save
    with open("seed_data.json", "w", encoding="utf-8") as f:
        json.dump(seeds, f, indent=2)
    print(f"Extracted: {len(seeds['programmes'])} progs, {len(seeds['facilities'])} facs, {len(seeds['staff'])} staff.")

if __name__ == "__main__":
    extract_seed_data()
