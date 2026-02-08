import os
from pymongo import MongoClient
from dotenv import load_dotenv
import re
import time

load_dotenv()

def log(msg):
    print(msg)
    with open("debug_output.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

MONGO_URI = os.getenv("MONGO_URI")

try:
    with open("debug_output.txt", "w", encoding="utf-8") as f:
        f.write("Starting DB Check...\n")

    start_time = time.time()
    log(f"Connecting to MongoDB...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client["UCSI_DB"]
    collection = db["UCSI_ MAJOR"]
    
    # Check connection
    client.server_info()
    elapsed = time.time() - start_time
    log(f"Connected in {elapsed:.2f} seconds")
    
    # Search for Formulation Science
    log("Searching for 'Formulation'...")
    query = {"Programme": {"$regex": "Formulation", "$options": "i"}}
    
    search_start = time.time()
    results = list(collection.find(query))
    search_elapsed = time.time() - search_start
    log(f"Search took {search_elapsed:.2f} seconds")
    
    if results:
        log(f"Found {len(results)} documents:")
        for r in results:
            log(f"- Programme: {r.get('Programme')}")
            log(f"  - URL: {r.get('Url')}")
            log(f"  - Duration: {r.get('Course Duration')}")
    else:
        log("No documents found for 'Formulation'.")

except Exception as e:
    log(f"Error: {e}")
