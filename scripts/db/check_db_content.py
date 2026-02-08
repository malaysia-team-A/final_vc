
import os
from pymongo import MongoClient
from dotenv import load_dotenv
import re

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    print("MONGO_URI not found in .env")
    exit(1)

try:
    print(f"Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client["UCSI_DB"]
    
    collection_name = "UCSI_ MAJOR" # Note the space
    collection = db[collection_name]
    
    print(f"Searching in collection: '{collection_name}'")
    
    # Search for Diploma in Opticianry
    query = {"Programme": {"$regex": "Opticianry", "$options": "i"}}
    results = list(collection.find(query))
    
    if results:
        print(f"Found {len(results)} documents:")
        for r in results:
            print(f"- {r.get('Programme')}: {r.get('URL', 'No URL found')}")
            # Print full doc to see what fields match
            print(r)
    else:
        print("No documents found for 'Opticianry'.")
        
except Exception as e:
    print(f"Error: {e}")
