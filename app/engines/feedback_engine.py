"""
Feedback Engine - User Feedback Collection System
Collects and stores user feedback on AI responses
"""
import json
import os
from datetime import datetime
from typing import Optional, Dict, List

FEEDBACK_FILE = "data/feedback_log.json"

class FeedbackEngine:
    def __init__(self, log_file: str = FEEDBACK_FILE):
        self.log_file = log_file
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Ensure the feedback log file exists"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump({"feedbacks": [], "stats": {"positive": 0, "negative": 0}}, f)
    
    def _load_data(self) -> Dict:
        """Load feedback data from file"""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"feedbacks": [], "stats": {"positive": 0, "negative": 0}}
    
    def _save_data(self, data: Dict):
        """Save feedback data to file"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def save_feedback(
        self,
        session_id: str,
        user_message: str,
        ai_response: str,
        rating: str,  # "positive" or "negative"
        comment: Optional[str] = None
    ) -> bool:
        """
        Save user feedback on an AI response
        
        Args:
            session_id: User session ID
            user_message: The original user question
            ai_response: The AI's response
            rating: "positive" (👍) or "negative" (👎)
            comment: Optional user comment
        
        Returns:
            bool: True if saved successfully
        """
        try:
            rating_norm = str(rating or "").strip().lower()
            if rating_norm not in {"positive", "negative"}:
                return False

            session_norm = str(session_id or "guest_session").strip() or "guest_session"
            user_message_norm = str(user_message or "").strip()
            ai_response_norm = str(ai_response or "").strip()
            comment_norm = None if comment is None else str(comment)

            if not user_message_norm or not ai_response_norm:
                return False

            data = self._load_data()
            
            feedback_entry = {
                "id": len(data["feedbacks"]) + 1,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_norm,
                "user_message": user_message_norm[:500],  # Limit length
                "ai_response": ai_response_norm[:1000],   # Limit length
                "rating": rating_norm,
                "comment": comment_norm
            }
            
            data["feedbacks"].append(feedback_entry)
            
            # Update stats
            if rating_norm == "positive":
                data["stats"]["positive"] += 1
            elif rating_norm == "negative":
                data["stats"]["negative"] += 1
            
            self._save_data(data)
            
            # Also save to MongoDB for persistence
            try:
                from .db_engine import db_engine
                if db_engine.connected:
                    db_engine.save_feedback(feedback_entry)
                    
                    # SELF-LEARNING: Positive Reinforcement
                    if rating_norm == "positive":
                        db_engine.save_learned_response(user_message_norm, ai_response_norm)
                    
                    # NEGATIVE LEARNING: Learn what NOT to say
                    elif rating_norm == "negative":
                        db_engine.save_bad_response(
                            query=user_message_norm,
                            bad_answer=ai_response_norm,
                            reason=comment_norm or "User marked as incorrect"
                        )
                        
            except Exception as db_err:
                print(f"MongoDB feedback save failed (non-critical): {db_err}")
            
            return True
            
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get feedback statistics"""
        data = self._load_data()
        stats = data.get("stats", {"positive": 0, "negative": 0})
        
        total = stats["positive"] + stats["negative"]
        if total > 0:
            stats["satisfaction_rate"] = round(stats["positive"] / total * 100, 1)
        else:
            stats["satisfaction_rate"] = 0
        
        stats["total_feedbacks"] = total
        return stats
    
    def get_recent_feedbacks(self, limit: int = 20) -> List[Dict]:
        """Get recent feedback entries"""
        data = self._load_data()
        feedbacks = data.get("feedbacks", [])
        return feedbacks[-limit:][::-1]  # Most recent first
    
    def get_negative_feedbacks(self, limit: int = 50) -> List[Dict]:
        """Get negative feedbacks for improvement analysis"""
        data = self._load_data()
        feedbacks = data.get("feedbacks", [])
        negative = [f for f in feedbacks if f.get("rating") == "negative"]
        return negative[-limit:][::-1]


# Singleton instance
feedback_engine = FeedbackEngine()


if __name__ == "__main__":
    # Test
    engine = FeedbackEngine()
    
    # Add test feedback
    engine.save_feedback(
        session_id="test_123",
        user_message="What is the gender ratio?",
        ai_response="The gender ratio is 51.6% female...",
        rating="positive"
    )
    
    print("Stats:", engine.get_stats())
