
import os
from pymongo import MongoClient, TEXT
from dotenv import load_dotenv

load_dotenv()
try:
    client = MongoClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=5000)
    db = client["UCSI_DB"]
    coll = db["UCSI_ MAJOR"]
    
    # Create Text Index
    print("Creating Text Index on 'Programme'...")
    coll.create_index([("Programme", TEXT)])
    print("Index Created Successfully!")
    
    # Test Search
    print("Testing $text search for 'Formulation Science'...")
    results = list(coll.find({"$text": {"$search": "Formulation Science"}}))
    print(f"Found {len(results)} results.")

except Exception as e:
    print(f"Error: {e}")
