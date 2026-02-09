"""
University Chatbot API - Main Server (Flask Version)
Features:
- AI-powered chatbot with LangChain + Local Ollama LLM
- JWT Authentication (OAuth2 Style)
- Dual Authentication for Sensitive Data (Grades)
- RAG (Retrieval-Augmented Generation)
- Log Anonymization
"""
from flask import Flask, request, jsonify, send_from_directory, Response
from app.engines.data_engine import DataEngine
from app.engines.ai_engine import AIEngine
from app.engines.feedback_engine import FeedbackEngine
import os
import json
import re
import logging
from datetime import datetime, timedelta
import secrets
from functools import wraps
from collections import deque

# Custom Modules
# Custom Modules
from app.utils import auth_utils
from app.utils import logging_utils
from app.engines.faq_cache_engine import faq_cache, unanswered_manager
from app.engines.language_engine import detect_language, get_localized_phrase, multilingual
from app.engines.rag_engine import rag_engine

# Setup Logging
logger = logging_utils.get_logger()

# Suppress noisy external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("werkzeug").setLevel(logging.WARNING) # Optional: cleaner flask logs

# Initialize Flask App
app = Flask(__name__, static_folder="static/site", static_url_path="/site")
app.secret_key = auth_utils.SECRET_KEY

# Initialize Engines
DATA_FILE = "data/Chatbot_TestData.xlsx" # Config artifact, logic moved to MongoDB
data_engine = DataEngine(DATA_FILE)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemma-3-12b-it")
ai_engine = AIEngine(MODEL_NAME)
feedback_engine = FeedbackEngine()

# In-memory storage for Dual Auth (High Security) sessions
# Format: { "student_number": datetime_expiry }
high_security_sessions = {}

# In-memory conversation history (prototype for follow-up context)
CONVERSATION_HISTORY_LIMIT = 12  # store last 6 exchanges
conversation_history_store = {}

# Ensure directories exist
if not os.path.exists("data/knowledge_base"):
    os.makedirs("data/knowledge_base")

PERSONAL_DATA_FIELDS = [
    "STUDENT_NUMBER",
    "STUDENT_ID",
    "STUDENT_NAME",
    "PREFERRED_NAME",
    "PROGRAMME_CODE",
    "PROGRAMME_NAME",
    "PROGRAMME",
    "PROFILE_STATUS",
    "PROFILE_TYPE",
    "ENROLLMENT_STATUS",
    "SEMESTER",
    "INTAKE",
    "CAMPUS",
    "FACULTY",
    "NATIONALITY",
    "GENDER",
    "EMAIL",
    "PHONE",
    "HOSTEL",
    "ADVISOR",
    "CURRENT_GPA",
    "CURRENT_CGPA",
    "GRADES",
    "LATEST_RESULTS",
    "DOB",
    "DATE_OF_BIRTH",
    "GPA",
    "CGPA"
]



def resolve_conversation_session(current_user, payload):
    """Return session key + conversation_id based on auth state."""
    if current_user and current_user.get("student_number"):
        student_number = current_user.get("student_number")
        return f"user:{student_number}", student_number, False

    payload_id = (payload or {}).get("conversation_id") if isinstance(payload, dict) else None
    if payload_id:
        return f"guest:{payload_id}", payload_id, False

    new_id = secrets.token_hex(8)
    return f"guest:{new_id}", new_id, True


def get_conversation_history(session_key):
    history = conversation_history_store.get(session_key)
    if not history:
        return []
    return list(history)


def append_conversation_message(session_key, role, content):
    if not session_key or not content:
        return
    history = conversation_history_store.setdefault(
        session_key,
        deque(maxlen=CONVERSATION_HISTORY_LIMIT)
    )
    history.append({"role": role, "content": content})

def is_grade_query(message):
    keywords = ['grade', 'result', 'exam', 'score', 'gpa', '성적', '점수', '학점', '결과']
    return any(k in message.lower() for k in keywords)

def check_personal_intent(message, search_term):
    # Check explicit AI hint
    if search_term and search_term.lower() in ["self", "me", "profile", "my info", "grades"]:
        return True
    
    # Check Keywords (English + Korean)
    keywords = ["my ", "i am", "who am i", "내 정보", "나의", "내 학번", "학생 정보", "성적"]
    if any(k in message.lower() for k in keywords):
        return True
        
    return False

def build_student_context(student_data):
    # Fetch extra info (Phase 2)
    semester_info = data_engine.get_semester_info(student_data.get("STUDENT_NUMBER"))
    
    context = "Here is the student's personal academic profile:\n"
    for k, v in student_data.items():
        if k not in ["_id", "PASSWORD", "password", "Password"]:
             context += f"- {k}: {v}\n"
             
    if semester_info:
        context += "\nCurrent Semester Details:\n"
        for k, v in semester_info.items():
            context += f"- {k}: {v}\n"
            
    return context


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # Check Authorization header (Bearer token)
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
        
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            payload = auth_utils.decode_access_token(token)
            if payload is None:
                return jsonify({'message': 'Token is invalid or expired!'}), 401
            current_user = payload
        except Exception as e:
            return jsonify({'message': 'Token is invalid!'}), 401
            
        return f(current_user, *args, **kwargs)
    
    return decorated

@app.route('/')
def home():
    return jsonify({"status": "University Chatbot API is running", "docs": "/site/code_hompage.html"})

@app.route('/site/<path:filename>')
def serve_static(filename):
    return send_from_directory('UI_hompage', filename)

# ===========================================
# AUTH ENDPOINTS
# ===========================================

@app.route('/api/login', methods=['POST'])
def login():
    """Login to get JWT Token"""
    try:
        data = request.get_json()
        student_number = data.get('student_number')
        name = data.get('name')
        
        # Verify user exists (Basic check for Name+ID match)
        is_valid, student_data, msg = data_engine.verify_student(student_number, name)
        
        if is_valid:
            # Generate JWT
            token = auth_utils.create_access_token({
                "student_number": student_number,
                "name": name,
                "role": "student"
            })
            
            logging_utils.log_audit("LOGIN", f"{name} ({student_number})", "Login successful")
            return jsonify({"success": True, "token": token, "user": {"name": name, "student_number": student_number}})
        else:
            logging_utils.log_audit("LOGIN_FAILED", f"{name} ({student_number})", f"Reason: {msg}")
            return jsonify({"success": False, "message": msg}), 401
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500

@app.route('/api/verify_password', methods=['POST'])
@token_required
def verify_high_security(current_user):
    """Verify password for Dual Auth"""
    try:
        data = request.get_json()
        password = data.get('password')
        student_number = current_user.get('student_number')
        
        # Get actual student record securely
        student = data_engine.get_student_info(student_number)
        
        if not student:
            return jsonify({"success": False, "message": "Student record not found"}), 404
            
        # Robust Password Retrieval (Case-insensitive)
        stored_password_hash = student.get("PASSWORD") or student.get("Password") or student.get("password") or ""

        # Verify
        if auth_utils.verify_password(password, stored_password_hash):
            # Grant high security access for 10 minutes
            high_security_sessions[student_number] = datetime.now() + timedelta(minutes=10)
            logging_utils.log_audit("HIGH_SECURITY_AUTH", student_number, "Password verification successful")
            return jsonify({"success": True, "message": "Identity verified. You can now access grades."})
        else:
            logging_utils.log_audit("HIGH_SECURITY_FAIL", student_number, "Password verification failed")
            return jsonify({"success": False, "message": "Incorrect password"}), 401
            
    except Exception as e:
        logger.error(f"Auth error: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500

# ===========================================
# CHAT ENDPOINTS
# ===========================================

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint supporting JWT and Dual Auth"""
    try:
        data = request.get_json() or {}
        user_message = data.get("message")
        
        # Get Token if available
        current_user = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1] if "Bearer " in request.headers['Authorization'] else None
            if token:
                current_user = auth_utils.decode_access_token(token)
        
        session_key, conversation_id, _ = resolve_conversation_session(current_user, data)
        conversation_history = get_conversation_history(session_key)

        if not user_message:
            return jsonify({"error": "Message is required", "conversation_id": conversation_id}), 400

        append_conversation_message(session_key, "user", user_message)
        
        # --- OPTIMIZED SINGLE-CALL FLOW ---
        
        # 0. Language Detection & FAQ Cache Check
        lang = detect_language(user_message)
        cached_result = faq_cache.get_cached_answer(user_message)
        
        if cached_result:
             # Return Cached Response
             response_payload = {
                 "text": cached_result["answer"],
                 "suggestions": cached_result.get("suggestions", [])
             }
             append_conversation_message(session_key, "assistant", json.dumps(response_payload))
             return jsonify({
                "response": json.dumps(response_payload),
                "session_id": conversation_id or secrets.token_hex(16),
                "type": "message",
                "user": current_user.get("name") if current_user else "Guest"
             })

        # 1. Intent Classification (의도 분류)
        intent_result = ai_engine.classify_intent(user_message)
        intent_category = intent_result.get("category", "faq")
        search_keywords = intent_result.get("search_keywords", [])
        logger.info(f"Intent: {intent_category}, Keywords: {search_keywords}")
        
        # 2. RAG Search (Vector Search for Q&A)
        rag_context = rag_engine.search(user_message)
        if rag_context:
            logger.info(f"RAG Context Found for: {user_message[:20]}...")
        
        # 3. Intent-Based DB Search (의도 기반 정형 데이터 조회)
        intent_context = ""
        try:
            from app.engines.db_engine import db_engine
            if db_engine.connected and db_engine.db is not None:
                # [ADDED] 의도에 따른 컬렉션 조회
                if intent_category == "facility":
                    facilities = list(db_engine.db['UCSI_FACILITY'].find({}, {"_id": 0}).limit(10))
                    if facilities:
                        intent_context = "[Facility Information]\n"
                        for f in facilities:
                            intent_context += f"- {f.get('name', '')}: {f.get('location', '')} ({f.get('opening_hours', '')})\n"
                
                elif intent_category == "staff":
                    staff_data = list(db_engine.db['UCSI_STAFF'].find({}, {"_id": 0}).limit(5))
                    if staff_data:
                        intent_context = "[Staff Directory]\n"
                        for dept in staff_data:
                            members = dept.get('staff_members', [])[:5]
                            for m in members:
                                intent_context += f"- {m.get('name', '')}: {m.get('role', '')} ({m.get('email', '')})\n"
                
                elif intent_category == "hostel":
                    hostels = list(db_engine.db['Hostel'].find({}, {"_id": 0}).limit(10))
                    if hostels:
                        intent_context = "[Hostel/Room Information]\n"
                        for h in hostels:
                            intent_context += f"- {h.get('room_type', 'Room')}: RM{h.get('rent_price', 'N/A')}/mo, Deposit: RM{h.get('deposit', 'N/A')} ({h.get('building', '')})\n"
                
                elif intent_category == "schedule":
                    schedules = list(db_engine.db['USCI_SCHEDUAL'].find({}, {"_id": 0}).limit(10))
                    if schedules:
                        intent_context = "[Academic Schedule]\n"
                        for s in schedules:
                            intent_context += f"- {s.get('event_name', '')}: {s.get('start_date', '')} ~ {s.get('end_date', '')}\n"
                
                elif intent_category == "major":
                    majors = list(db_engine.db['UCSI_ MAJOR'].find({}, {"_id": 0}).limit(10))
                    if majors:
                        intent_context = "[Programme/Major Information]\n"
                        for m in majors:
                            intent_context += f"- {m.get('Programme', '')}: RM{m.get('Local Students Fees', 'N/A')} ({m.get('Course Duration', '')})\n"

                elif intent_category == "personal":
                    # [ADDED] Pre-fetch personal data if logged in
                    if current_user:
                        student_data = db_engine.get_student_by_number(current_user.get("student_number"))
                        if student_data:
                            # Use existing helper to format context
                            intent_context = "[Personal Student Record]\n" + build_student_context(student_data)
                    else:
                        # Let AI know status, but don't force login response yet (AI might just be chatting)
                        # AI typically handles "needs_context" -> "login_hint" flow, but context helps.
                        intent_context = "[System Note] User is NOT logged in. If they ask for personal data, ask them to login."
                
        except Exception as e:
            logger.error(f"Intent-based DB search error: {e}")
        
        # 4. Combine Contexts (RAG + Intent-Based)
        combined_context = ""
        if rag_context:
            combined_context = rag_context
        if intent_context:
            combined_context = (combined_context + "\n\n" + intent_context) if combined_context else intent_context
        
        # 5. AI Processing with Combined Context
        initial_result = ai_engine.process_message(
            user_message, 
            data_context=combined_context, 
            conversation_history=list(conversation_history),
            language=lang
        )
        
        response_payload = {}
        response_text = ""
        context_used = combined_context # Track what context was used

        # 3. Check if *Personal* Data is needed (and not provided by RAG)
        # If AI says "needs_context" even after providing RAG (or if RAG was empty),
        # it might be asking for personal student data (grades, profile).
        if initial_result.get("needs_context"):
            try:
                # Only fetch personal data if RAG didn't answer it (or if AI specifically requested personal info)
                # We check intent again to be safe.
                search_term = initial_result.get("search_term")
                
                # A. Check for Personal Data / Grades first (Security)
                is_personal = check_personal_intent(user_message, search_term)
                if is_personal:
                     if not current_user:
                        return jsonify({
                            "response": "🔒 Please login to access personal information.",
                            "type": "login_hint",
                            "conversation_id": conversation_id
                        })
                     # Dual Auth Check for Grades
                     if is_grade_query(user_message):
                        expiry = high_security_sessions.get(current_user.get("student_number"))
                        if not expiry or datetime.now() > expiry:
                            return jsonify({
                                "response": "🔒 Security Check: Please enter your password to view examination results.",
                                "type": "password_prompt",
                                "conversation_id": conversation_id
                            })
                            
                     student_data = data_engine.get_student_info(current_user.get("student_number"))
                     if student_data:
                         personal_context = build_student_context(student_data)
                         # Combine RAG context with Personal Context if needed
                         context_used = (rag_context + "\n\n" + personal_context) if rag_context else personal_context
                     else:
                         context_used = "Student record not found."
                    
                     # Re-process with Personal Data
                     final_result = ai_engine.process_message(
                        user_message, 
                        data_context=context_used, 
                        conversation_history=list(conversation_history),
                        language=lang
                    )
                     # Update result
                     initial_result = final_result

                # B. If not personal and RAG failed, try structured DB stats
                elif not rag_context and ("count" in user_message.lower() or "how many" in user_message.lower()):
                    stats_context = data_engine.get_summary_stats()
                    # Re-process
                    final_result = ai_engine.process_message(
                        user_message, 
                        data_context=str(stats_context), 
                        conversation_history=list(conversation_history),
                        language=lang
                    )
                    initial_result = final_result

            except Exception as e:
                logger.error(f"Context Fetch Error: {e}")
                response_text = "I encountered an error looking up that information."
                response_payload = {"text": response_text, "suggestions": []}

        # 4. Final Response Construction (Common Logic)
        if not response_payload:
            response_text = initial_result.get("response", "I couldn't find that info.")
            # Remove markdown formatting
            response_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', response_text)  # **bold**
            response_text = re.sub(r'\*([^*]+)\*', r'\1', response_text)  # *italic*
            response_text = re.sub(r'^#+\s*', '', response_text, flags=re.MULTILINE)  # ## headers
            
            response_payload = {
                "text": response_text,
                "suggestions": initial_result.get("suggestions", [])
            }

        # Update History
        append_conversation_message(session_key, "assistant", json.dumps(response_payload))

        # Return structured JSON for frontend
        # format: { response: JSON_STRING, session_id: STR }
        return jsonify({
            "response": json.dumps(response_payload),
            "session_id": conversation_id,
            "type": "message",
            "user": current_user.get("name") if current_user else "Guest"
        })

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500

# ===========================================
# FEEDBACK & ADMIN
# ===========================================
# FEEDBACK & ADMIN
# ===========================================

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        logging_utils.log_audit("FEEDBACK", "User", f"Rating: {data.get('rating')}")
        feedback_engine.save_feedback(
            session_id="jwt_session", # Simplified for now
            user_message=data.get("user_message"),
            ai_response=data.get("ai_response"),
            rating=data.get("rating"),
            comment=data.get("comment")
        )
        if data.get("rating") == "positive":
            # POSITIVE FEEDBACK: Reinforcement Learning
            # If user likes the answer, consider it a candidate for FAQ
            try:
                question = data.get("user_message")
                answer_json = data.get("ai_response") # Expecting JSON string
                
                # Extract text from JSON response if needed
                answer_text = answer_json
                suggestions = []
                try:
                    parsed = json.loads(answer_json)
                    if isinstance(parsed, dict):
                        answer_text = parsed.get("text", answer_json)
                        suggestions = parsed.get("suggestions", [])
                except:
                    pass
                
                # Boost frequency or auto-add
                # For this implementation, we try to auto-cache
                added = faq_cache.auto_cache_if_needed(question, answer_text, suggestions)
                if added:
                    logger.info(f"Auto-cached FAQ due to positive feedback: {question}")
            except Exception as e:
                logger.error(f"Error learning from positive feedback: {e}")

        elif data.get("rating") == "negative":
            # NEGATIVE FEEDBACK: Log for review
            try:
                unanswered_manager.log_unanswered(
                    question=data.get("user_message") or "",
                    reason="negative_feedback",
                    context=data.get("comment") or ""
                )
            except Exception as err:
                logger.warning(f"Failed to log negative feedback: {err}")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout (Client should also delete token)"""
    # For stateless JWT, we can't really invalidate unless we blacklist. 
    # For this demo, we just log it.
    logging_utils.log_audit("LOGOUT", "User", "Logout request")
    return jsonify({"success": True})

@app.route('/api/export_chat', methods=['GET'])
@token_required
def export_chat(current_user):
    """Export conversation history for the current session"""
    try:
        session_id = request.args.get('session_id')
        # If no session_id provided, try to find one or return empty/error
        # For simplicity, if not provided, we return just a header
        
        history = user_sessions.get(session_id, []) if session_id else []
        
        # Format as text
        export_text = f"Chat History for {current_user.get('name')} ({current_user.get('student_number')})\n"
        export_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        export_text += "="*40 + "\n\n"
        
        for msg in history:
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')
            export_text += f"[{role}]: {content}\n\n"
            
        # Return as downloadable file
        return Response(
            export_text,
            mimetype="text/plain",
            headers={"Content-disposition": f"attachment; filename=chat_history_{current_user.get('student_number')}.txt"}
        )
            
    except Exception as e:
        logger.error(f"Export error: {e}")
        return jsonify({"error": str(e)}), 500

# ===========================================
# ADMIN ENDPOINTS
# ===========================================

@app.route('/admin')
def admin_page():
    """Serve Admin Dashboard"""
    if os.path.exists("static/admin/admin.html"):
        return send_from_directory('static/admin', 'admin.html')
    return "Admin panel not found", 404

@app.route('/api/admin/stats', methods=['GET'])
def get_admin_stats():
    """Get statistics for admin dashboard"""
    try:
        # Get Feedback Stats
        feedback_stats = feedback_engine.get_stats()
        recent_feedbacks = feedback_engine.get_recent_feedbacks(limit=10)
        
        # Get Learning Stats
        unanswered = unanswered_manager.get_unresolved()
        
        return jsonify({
            "satisfaction_rate": feedback_stats.get("satisfaction_rate", 0),
            "total_feedbacks": feedback_stats.get("total_feedbacks", 0),
            "unanswered_count": len(unanswered),
            "unanswered_logs": unanswered[-10:], # Last 10
            "recent_feedbacks": recent_feedbacks
        })
    except Exception as e:
        logger.error(f"Admin stats error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/upload', methods=['POST'])
def upload_document():
    """Upload a document to the knowledge base"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "message": "No file part"}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"success": False, "message": "No selected file"}), 400
            
        if file:
            # Ensure directory exists
            if not os.path.exists("data/knowledge_base"):
                os.makedirs("data/knowledge_base")
                
            file_path = f"data/knowledge_base/{file.filename}"
            file.save(file_path)
            
            # Ingest into RAG
            from app.engines.rag_engine import rag_engine
            success = rag_engine.ingest_file(file_path)
            
            if success:
                return jsonify({"success": True, "message": f"Successfully ingested {file.filename}"})
            else:
                return jsonify({"success": False, "message": "File saved but failed to ingest into Vector DB"})
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/admin/files', methods=['GET'])
def list_files():
    """List files in knowledge base"""
    try:
        files = []
        if os.path.exists("data/knowledge_base"):
            for f in os.listdir("data/knowledge_base"):
                if f.endswith(('.pdf', '.txt', '.csv', '.docx')):
                    files.append(f)
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/files', methods=['DELETE'])
def delete_file():
    """Delete a file from knowledge base"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        if not filename:
            return jsonify({"success": False, "message": "Filename required"}), 400
        
        file_path = os.path.join("data/knowledge_base", filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            # Note: Ideally we should re-index RAG here, but for now we just delete source
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "message": "File not found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    try:
        # Setup File Logging
        file_handler = logging.FileHandler('server.log')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        print(f"Starting Flask server with Google GenAI Native Model: {MODEL_NAME}")
        # Disable reloader to prevent double-execution/subprocess issues in certain envs
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    except Exception as e:
        with open("crash_log.txt", "w") as f:
            f.write(f"Server Crashed: {str(e)}")
        print(f"Server Crashed: {e}")

