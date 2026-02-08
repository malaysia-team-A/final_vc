
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
try:
    client = MongoClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=5000)
    db = client["UCSI_DB"]
    coll = db["UCSI_ MAJOR"]
    
    print("--- Current Indexes ---")
    for name, idx in coll.index_information().items():
        print(f"Name: {name}, Keys: {idx['key']}")
        
except Exception as e:
    print(e)
