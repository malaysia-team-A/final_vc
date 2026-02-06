"""
Data Engine - Student Data Access Layer
Now uses MongoDB instead of Excel
"""
import pandas as pd
from .db_engine import db_engine
import json
import re

class DataEngine:
    def __init__(self, excel_path=None):
        """
        Initialize DataEngine with MongoDB backend
        excel_path parameter kept for backward compatibility but not used
        """
        self.db = db_engine
        print(f"DataEngine initialized with MongoDB (connected: {self.db.connected})")

    def get_column_names(self):
        """Return available field names from MongoDB"""
        if not self.db.connected:
            return []
        
        # Get a sample document to extract field names
        students = self.db.get_all_students()
        if students and len(students) > 0:
            return list(students[0].keys())
        return []

    def verify_student(self, student_number, name):
        """
        Verify a student exists with matching student number and name
        Returns (is_valid, student_data, message) tuple
        """
        if not self.db.connected:
            return (False, None, "Database connection error")
        
        print(f"DEBUG: Verifying student_number='{student_number}', name='{name}'")
        
        # Get student by student number
        student = self.db.get_student_by_number(student_number)
        
        if not student:
            print(f"DEBUG: Student number '{student_number}' not found in DB")
            return (False, None, "Student number not found")
        
        # Verify name matches (case-insensitive)
        # Check all possible variations of field names
        student_name = student.get("STUDENT_NAME") or student.get("name") or student.get("Name") or student.get("FullName")
        
        print(f"DEBUG: Found student record: {student}")
        
        if not student_name:
            print("DEBUG: Student record found but no name field found")
            return (False, None, "Student record corrupted (no name field)")

        student_name_str = str(student_name).strip().lower()
        input_name = str(name).strip().lower()
        
        if student_name_str == input_name:
            print("DEBUG: Verification successful")
            return (True, student, "Verification successful")
        else:
            print(f"DEBUG: Name mismatch. DB='{student_name_str}', Input='{input_name}'")
            return (False, None, "Name does not match student number")

    def get_student_info(self, student_number):
        """Get a specific student's information by student number"""
        if not self.db.connected:
            return None
        
        return self.db.get_student_by_number(student_number)

    def get_summary_stats(self):
        """Get general statistics (non-sensitive)"""
        if not self.db.connected:
            return {
                "total_students": 0,
                "gender_breakdown": {},
                "nationality_breakdown": {},
                "message": "Data currently unavailable (DB Disconnected)"
            }
        
        stats = self.db.get_student_stats()
        
        # Format for compatibility with existing code
        formatted_stats = {
            "total_students": stats.get("total_students", 0),
            "gender_breakdown": stats.get("gender", {}),
            "nationality_breakdown": stats.get("top_nationalities", {}),
            "columns": ["STUDENT_NUMBER", "STUDENT_NAME", "NATIONALITY", "GENDER", 
                       "PROGRAMME_CODE", "PROGRAMME_NAME", "PROFILE_STATUS", 
                       "PROFILE_TYPE", "INTAKE"]
        }
        
        return formatted_stats

    def search_programme_info(self, user_message):
        """Search MongoDB for programme/program keywords"""
        if not self.db.connected or not user_message:
            return []

        tokens = [tok for tok in re.split(r'[^A-Za-z0-9]+', str(user_message)) if len(tok) > 3]
        if not tokens:
            tokens = [str(user_message)]
        return self.db.search_programme_by_keywords(tokens)

    def search_students(self, query):
        """Search students (limit fields for privacy)"""
        if not self.db.connected:
            return []
        
        # Search by name
        student = self.db.get_student_by_name(query)
        if student:
            return [student]
        return []

    def get_semester_info(self, student_number):
        """Get semester info wrapper"""
        if not self.db.connected: return {}
        return self.db.get_semester_info(student_number)

    def search_staff(self, query):
        """Search staff wrapper"""
        if not self.db.connected: return []
        return self.db.search_staff(query)


if __name__ == "__main__":
    engine = DataEngine()
    print("\n=== Summary Stats ===")
    stats = engine.get_summary_stats()
    print(f"Total Students: {stats.get('total_students')}")
    print(f"Gender Breakdown: {stats.get('gender_breakdown')}")
    print("\n=== Columns ===")
    print(engine.get_column_names())
