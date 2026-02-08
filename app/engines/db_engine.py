"""
Database Engine - MongoDB Atlas Connection Manager
Handles all MongoDB operations for the UCSI Chatbot
"""
import os
from typing import Optional, Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Conditional import
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    HAS_PYMONGO = True
except ImportError:
    HAS_PYMONGO = False
    print("Warning: pymongo not installed. Run: pip install pymongo")

class DatabaseEngine:
    def __init__(self):
        self.client = None
        self.db = None
        self.connected = False
        self.student_collection_name = "UCSI"  # Default
        
        if HAS_PYMONGO:
            self._connect()
    
    def _connect(self):
        """Establish connection to MongoDB Atlas with timeout"""
        try:
            # Load environment variables
            if not os.getenv("MONGO_URI"):
                load_dotenv()
            if not os.getenv("MONGO_URI"):
                load_dotenv()
            
            uri = os.getenv("MONGO_URI")
            if not uri:
                print("Error: MONGO_URI not found in .env")
                return

            # Add timeout to prevent hanging
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            
            # Verify connection
            self.client.admin.command('ping')
            
            # Get Database
            db_name = uri.split('/')[-1].split('?')[0] or "UCSI_DB"
            self.db = self.client[db_name]
            self.connected = True
            print(f"[DB] Successfully connected to MongoDB: {db_name}")

            # Smart detection of Student Collection
            try:
                colls = self.db.list_collection_names()
                candidates = ["UCSI", "students", "Students", "UCSI_STUDENTS"]
                found = False
                # First check if default candidates exist
                for c in candidates:
                    if c in colls:
                        self.student_collection_name = c
                        found = True
                        break
                
                # If not found, look for any collection with 'student' in name
                if not found:
                    for c in colls:
                        if "student" in c.lower() or "ucsi" in c.lower():
                            self.student_collection_name = c
                            found = True
                            print(f"DEBUG: Auto-detected student collection: {c}")
                            break
                            
                # Validate by checking if it has STUDENT_NUMBER
                if found or colls:
                    target = self.student_collection_name if found else colls[0]
                    sample = self.db[target].find_one()
                    if sample:
                        print(f"DEBUG: Using collection '{target}' for student data.")
                        self.student_collection_name = target
            except Exception as e:
                print(f"Warning: Could not auto-detect collections: {e}")
            
        except Exception as e:
            print(f"[DB][ERROR] Database connection failed: {e}")
            self.connected = False
    
    @property
    def student_coll(self):
        """Helper to get the student collection object"""
        if self.db is not None:
            return self.db[self.student_collection_name]
        return None

    # ===========================================
    # STUDENTS COLLECTION
    # ===========================================
    
    def get_student_by_number(self, student_number: str) -> Optional[Dict]:
        """Find student by student number (handles string and int)"""
        if self.student_coll is None or not self.connected:
            return None
        
        try:
            # Try string
            student = self.student_coll.find_one({"STUDENT_NUMBER": str(student_number)})
            if student: return student
            
            # Try int
            if str(student_number).isdigit():
                student = self.student_coll.find_one({"STUDENT_NUMBER": int(student_number)})
                if student: return student
                
            # Try fuzzy field names if standard query failed
            id_fields = ["student_number", "StudentNumber", "student_id", "id", "ID"]
            for field in id_fields:
                s = self.student_coll.find_one({field: str(student_number)})
                if s: return s
                if str(student_number).isdigit():
                    s = self.student_coll.find_one({field: int(student_number)})
                    if s: return s
                    
            return None
        except Exception as e:
            print(f"DB Error: {e}")
            return None
    
    def get_student_by_name(self, name: str) -> Optional[Dict]:
        """Find student by name (case-insensitive)"""
        if self.student_coll is None or not self.connected:
            return None
        try:
            # Try standard field
            res = self.student_coll.find_one({
                "STUDENT_NAME": {"$regex": f"^{name}$", "$options": "i"}
            })
            if res: return res
            
            # Try common variations
            name_fields = ["name", "Name", "StudentName", "full_name"]
            for field in name_fields:
                res = self.student_coll.find_one({
                    field: {"$regex": f"^{name}$", "$options": "i"}
                })
                if res: return res
            return None
        except Exception as e:
            print(f"DB Error: {e}")
            return None
    
    def get_all_students(self) -> List[Dict]:
        """Get all students"""
        if self.student_coll is None or not self.connected:
            return []
        try:
            return list(self.student_coll.find({}, {"_id": 0}))
        except Exception as e:
            print(f"DB Error: {e}")
            return []
    
    def search_programme_by_keywords(self, keywords: List[str], limit: int = 5) -> List[Dict]:
        """Find programme information by keyword list"""
        if self.student_coll is None or not self.connected:
            return []

        sanitized = []
        for kw in keywords:
            if not kw:
                continue
            kw_clean = str(kw).strip()
            if len(kw_clean) < 3:
                continue
            sanitized.append(kw_clean)
        if not sanitized:
            return []

        or_conditions = []
        fields = ["PROGRAMME_NAME", "PROGRAMME", "PROGRAMME_TITLE", "PROGRAMME_NAME_FULL"]
        for kw in sanitized:
            regex = {"$regex": kw, "$options": "i"}
            for field in fields:
                or_conditions.append({field: regex})
        if not or_conditions:
            return []

        projection = {
            "PROGRAMME_NAME": 1,
            "PROGRAMME": 1,
            "PROGRAMME_TITLE": 1,
            "PROGRAMME_CODE": 1,
            "PROGRAMME_LEVEL": 1,
            "LEVEL": 1,
            "FACULTY": 1,
            "FACULTY_NAME": 1,
            "SCHOOL": 1,
            "CAMPUS": 1,
            "INTAKE": 1,
            "PROFILE_TYPE": 1,
            "PROGRAMME_FACULTY": 1
        }

        try:
            cursor = self.student_coll.find({"$or": or_conditions}, projection).limit(limit * 3)
            results = []
            seen = set()
            for doc in cursor:
                programme_name = (
                    doc.get("PROGRAMME_NAME") or doc.get("PROGRAMME") or doc.get("PROGRAMME_TITLE")
                )
                key = (programme_name or "").strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                entry = {
                    "programme_name": programme_name,
                    "programme_code": doc.get("PROGRAMME_CODE"),
                    "faculty": doc.get("FACULTY") or doc.get("FACULTY_NAME") or doc.get("PROGRAMME_FACULTY"),
                    "campus": doc.get("CAMPUS"),
                    "level": doc.get("PROGRAMME_LEVEL") or doc.get("LEVEL"),
                    "profile_type": doc.get("PROFILE_TYPE"),
                    "intake": doc.get("INTAKE")
                }
                results.append(entry)
                if len(results) >= limit:
                    break
            return results
        except Exception as e:
            print(f"DB Error: {e}")
            return []

    def get_semester_info(self, student_number: str) -> Dict:
        """Get semester and enrollment status"""
        student = self.get_student_by_number(student_number)
        if not student:
            return {}
        
        # Extract meaningful status
        return {
            "profile_status": student.get("PROFILE_STATUS", "Unknown"),
            "intake": student.get("INTAKE", "Unknown"),
            "current_semester": student.get("SEMESTER", "N/A"), # Assuming field exists or N/A
            "programme": student.get("PROGRAMME_CODE"),
            "enrollment_status": "Active" if "active" in str(student.get("PROFILE_STATUS", "")).lower() else "Inactive"
        }
        
    def search_staff(self, query: str, limit: int = 5) -> List[Dict]:
        """Search staff directory"""
        if self.db is None or not self.connected:
            return []

        colls = self.db.list_collection_names()
        staff_coll = self.db["UCSI_STAFF"] if "UCSI_STAFF" in colls else None

        if staff_coll is None:
            # Try finding a collection with 'staff' in it
            for c in colls:
                if "staff" in c.lower() or "faculty" in c.lower():
                    staff_coll = self.db[c]
                    break

        if staff_coll is None:
            return []
            
        try:
            # Match actual UCSI_STAFF schema: major + nested staff_members[]
            regex = {"$regex": query, "$options": "i"}
            cursor = staff_coll.find({
                "$or": [
                    {"major": regex},
                    {"staff_members.name": regex},
                    {"staff_members.role": regex},
                    {"staff_members.email": regex},
                    # Backward-compat fields for legacy data
                    {"NAME": regex}, 
                    {"name": regex},
                    {"DEPARTMENT": regex},
                    {"POSITION": regex}
                ]
            }, {"_id": 0}).limit(limit)
            
            return list(cursor)
        except Exception as e:
            print(f"Staff Search Error: {e}")
            return []

    def get_student_stats(self) -> Dict:
        """Aggregate high-level student statistics from UCSI collection."""
        if self.student_coll is None or not self.connected:
            return {"total_students": 0, "gender": {}, "top_nationalities": {}}

        try:
            total_students = self.student_coll.count_documents({})

            gender_pipeline = [
                {"$group": {"_id": "$GENDER", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            nationality_pipeline = [
                {"$group": {"_id": "$NATIONALITY", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 10}
            ]

            gender_rows = list(self.student_coll.aggregate(gender_pipeline))
            nationality_rows = list(self.student_coll.aggregate(nationality_pipeline))

            gender = {str(r.get("_id", "Unknown")): r.get("count", 0) for r in gender_rows}
            top_nationalities = {str(r.get("_id", "Unknown")): r.get("count", 0) for r in nationality_rows}

            return {
                "total_students": total_students,
                "gender": gender,
                "top_nationalities": top_nationalities
            }
        except Exception as e:
            print(f"Student stats error: {e}")
            return {"total_students": 0, "gender": {}, "top_nationalities": {}}

    # ===========================================
    # FEEDBACKS & LOGS (Using separate collections)
    # ===========================================
    
    def save_feedback(self, feedback_data: Dict) -> bool:
        if self.db is None or not self.connected: return False
        try:
            # Correct Collection Name: 'Feedback'
            self.db.Feedback.insert_one(feedback_data)
            return True
        except: return False

    def get_feedback_stats(self) -> Dict:
        if self.db is None or not self.connected: return {"total": 0}
        try:
            return {
                "total": self.db.Feedback.count_documents({}),
                "positive": self.db.Feedback.count_documents({"rating": "positive"}),
                "negative": self.db.Feedback.count_documents({"rating": "negative"})
            }
        except: return {"total": 0}

    def log_unanswered(self, question_data: Dict) -> bool:
        if self.db is None or not self.connected: return False
        try:
            self.db.unanswered.insert_one(question_data)
            return True
        except: return False

    def check_negative_feedback(self, query: str) -> bool:
        """Check if similar query has negative feedback"""
        if self.db is None or not self.connected: return False
        try:
            # Simple check: exact string match or containment?
            # ideally use text search or vector, but for now exact/partial string
            # Check last 50 negative feedbacks
            negatives = list(self.db.Feedback.find({"rating": "negative"}).sort("_id", -1).limit(50))
            for n in negatives:
                if n.get("user_input", "").strip().lower() == query.strip().lower():
                    return True
            return False
        except: return False

    # ===========================================
    # SELF-LEARNING (Knowledge Base)
    # ===========================================
    def save_learned_response(self, query: str, answer: str) -> bool:
        """Save a confirmed high-quality Q&A pair."""
        if self.db is None or not self.connected: return False
        try:
            # Upsert based on query to update answer if it changes
            self.db.LearnedQA.update_one(
                {"query": query.strip().lower()},
                {"$set": {
                    "query": query.strip().lower(),
                    "answer": answer,
                    "timestamp": datetime.now(),
                    "source": "user_feedback"
                }},
                upsert=True
            )
            # Create Index if not exists
            self.db.LearnedQA.create_index([("query", "text")])
            return True
        except Exception as e:
            print(f"Error saving learned response: {e}")
            return False

    def search_learned_response(self, query: str) -> str:
        """Search for a learned response to the query."""
        if self.db is None or not self.connected: return None
        try:
            # 1. Exact Match (High Priority)
            exact = self.db.LearnedQA.find_one({"query": query.strip().lower()})
            if exact:
                return exact.get("answer")
            
            # 2. Text Search (Medium Priority)
            # Find best text match with high score
            results = list(self.db.LearnedQA.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(1))
            
            if results and results[0].get('score', 0) > 1.5: # Threshold for learned relevance
                return results[0].get("answer")
            
            return None
        except: return None
    
    # ===========================================
    # NEGATIVE LEARNING (Bad Responses to Avoid)
    # ===========================================
    def save_bad_response(self, query: str, bad_answer: str, reason: str = "") -> bool:
        """Save a negative feedback response - 'Do NOT answer like this'."""
        if self.db is None or not self.connected: return False
        try:
            q = str(query or "").strip().lower()
            if not q:
                return False
            self.db.BadQA.update_one(
                {"query": q},
                {
                    "$set": {
                        "query": q,
                        "bad_answer": str(bad_answer or ""),
                        "reason": str(reason or ""),
                        "timestamp": datetime.now(),
                    },
                    "$inc": {"count": 1},
                },
                upsert=True
            )
            # Create Index if not exists
            self.db.BadQA.create_index([("query", "text")])
            print(f"[DB] Saved bad response for: {q[:50]}...")
            return True
        except Exception as e:
            print(f"Error saving bad response: {e}")
            return False
    
    def search_bad_responses(self, query: str) -> list:
        """Search for known bad responses for this query."""
        if self.db is None or not self.connected: return []
        try:
            results = []
            
            # 1. Exact Match
            exact = self.db.BadQA.find_one({"query": query.strip().lower()})
            if exact:
                results.append({
                    "bad_answer": exact.get("bad_answer", ""),
                    "reason": exact.get("reason", "User marked as incorrect"),
                    "count": exact.get("count", 1)
                })
            
            # 2. Fuzzy Text Search
            try:
                cursor = self.db.BadQA.find(
                    {"$text": {"$search": query}},
                    {"score": {"$meta": "textScore"}}
                ).sort([("score", {"$meta": "textScore"})]).limit(3)
                
                for doc in cursor:
                    if doc.get('score', 0) > 1.0:
                        results.append({
                            "bad_answer": doc.get("bad_answer", ""),
                            "reason": doc.get("reason", "Similar query had issues"),
                            "count": doc.get("count", 1)
                        })
            except:
                pass
            
            return results
        except:
            return []
    
    def get_unanswered_questions(self, limit: int = 50) -> List[Dict]:
        if self.db is None or not self.connected: return []
        try:
            return list(self.db.unanswered.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
        except: return []

# Singleton instance
db_engine = DatabaseEngine()

if __name__ == "__main__":
    if db_engine.connected:
        print("Database connection test successful!")
        print(f"Collections: {db_engine.db.list_collection_names()}")
    else:
        print("Database not connected. Check your .env file and MongoDB Atlas settings.")
