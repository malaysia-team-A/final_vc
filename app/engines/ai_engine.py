import os
import re
import json
import ast
import time
import random
import threading
from datetime import datetime, timedelta
from google import genai # New SDK
from .language_engine import multilingual  # Multi-language support (Updated relative import)

# ============================================================
# 🚀 ENTERPRISE-GRADE API RESILIENCE CONFIGURATION
# Inspired by Netflix Hystrix, Google SRE, AWS patterns
# ============================================================

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


# Adaptive Exponential Backoff Configuration
MAX_RETRIES = _env_int("AI_MAX_RETRIES", 2)
BASE_DELAY = _env_float("AI_BASE_DELAY", 0.5)  # Starting delay in seconds
MAX_DELAY = _env_float("AI_MAX_DELAY", 2.0)  # Maximum delay cap
JITTER_FACTOR = _env_float("AI_JITTER_FACTOR", 0.2)  # Random jitter to prevent thundering herd

# Circuit Breaker Configuration (Netflix Hystrix pattern)
CIRCUIT_FAILURE_THRESHOLD = _env_int("AI_CIRCUIT_FAILURE_THRESHOLD", 8)  # Open circuit after N consecutive failures
CIRCUIT_RECOVERY_TIMEOUT = _env_int("AI_CIRCUIT_RECOVERY_TIMEOUT", 8)  # Seconds before attempting recovery
CIRCUIT_HALF_OPEN_MAX_CALLS = _env_int("AI_CIRCUIT_HALF_OPEN_MAX_CALLS", 3)  # Test calls when half-open

# Token Bucket Rate Limiter (Google-style throttling)
RATE_LIMIT_RPM = _env_int("AI_RATE_LIMIT_RPM", 120)  # Requests per minute
RATE_LIMIT_TOKENS = _env_int("AI_RATE_LIMIT_TOKENS", max(30, RATE_LIMIT_RPM))  # Bucket capacity
RATE_LIMIT_REFILL_RATE = max(_env_float("AI_RATE_LIMIT_REFILL_RATE", RATE_LIMIT_RPM / 60.0), 0.1)  # Tokens per second


class CircuitBreaker:
    """
    Circuit Breaker Pattern Implementation
    States: CLOSED (normal) -> OPEN (failing) -> HALF_OPEN (testing)
    """
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"
    
    def __init__(self, failure_threshold=CIRCUIT_FAILURE_THRESHOLD, 
                 recovery_timeout=CIRCUIT_RECOVERY_TIMEOUT):
        self.state = self.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = None
        self.half_open_calls = 0
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        """Check if request should be allowed"""
        with self._lock:
            if self.state == self.CLOSED:
                return True
            elif self.state == self.OPEN:
                # Check if recovery timeout has passed
                if self.last_failure_time and \
                   datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                    self.state = self.HALF_OPEN
                    self.half_open_calls = 0
                    print(f"[CircuitBreaker] Transitioning to HALF_OPEN state")
                    return True
                return False
            else:  # HALF_OPEN
                if self.half_open_calls < CIRCUIT_HALF_OPEN_MAX_CALLS:
                    self.half_open_calls += 1
                    return True
                return False
    
    def record_success(self):
        """Record successful call"""
        with self._lock:
            if self.state == self.HALF_OPEN:
                print(f"[CircuitBreaker] Recovery successful, closing circuit")
            self.state = self.CLOSED
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed call"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == self.HALF_OPEN:
                self.state = self.OPEN
                print(f"[CircuitBreaker] Recovery failed, reopening circuit")
            elif self.failure_count >= self.failure_threshold:
                self.state = self.OPEN
                print(f"[CircuitBreaker] Threshold reached ({self.failure_count}), opening circuit for {self.recovery_timeout}s")
    
    def get_status(self) -> dict:
        """Get circuit breaker status for monitoring"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class TokenBucketRateLimiter:
    """
    Token Bucket Rate Limiter (Google Cloud-style)
    Provides smooth rate limiting with burst capacity
    """
    def __init__(self, capacity=RATE_LIMIT_TOKENS, refill_rate=RATE_LIMIT_REFILL_RATE):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self._lock = threading.Lock()
    
    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def acquire(self, timeout=10.0) -> bool:
        """
        Try to acquire a token, blocking up to timeout seconds
        Returns True if token acquired, False otherwise
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
            
            # Wait a bit before retrying
            time.sleep(0.1)
        
        return False
    
    def get_status(self) -> dict:
        """Get rate limiter status for monitoring"""
        with self._lock:
            self._refill()
            return {
                "available_tokens": round(self.tokens, 2),
                "capacity": self.capacity,
                "refill_rate": f"{self.refill_rate}/sec"
            }


# Global instances for singleton pattern
_circuit_breaker = CircuitBreaker()
_rate_limiter = TokenBucketRateLimiter()

class AIEngine:
    def __init__(self, model_name="gemma-3-27b-it"):
        """
        Initialize using the NEW Google Gen AI SDK (google-genai)
        """
        self.raw_model_name = model_name
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.client = None

        if self.api_key:
            try:
                # New SDK Client Initialization
                print(f"[INIT] Initializing Gemini AI ({self.raw_model_name}) via NEW Google Gen AI SDK...")
                self.client = genai.Client(api_key=self.api_key)
                
                # Normalize model name for new SDK (e.g., remove 'models/' prefix if present)
                # The new SDK typically expects 'gemini-1.5-flash'
                self.model_name = self.raw_model_name.replace("models/", "")
                
            except Exception as e:
                print(f"Gemini Init Failed: {e}")
                self.client = None
        else:
            print("[ERROR] GOOGLE_API_KEY not found.")

        # PROMPTS (Modified dynamically in process_message)
        
        self.qa_template = """You are UCSI buddy (name: Buddy), a friendly UCSI University assistant.

Context:
{context}

Conversation History:
{conversation}

Question: {question}

=== ABSOLUTE RULES (NEVER BREAK THESE) ===

1. [NO_RELEVANT_DATA_FOUND] RULE:
   - If Context contains "[NO_RELEVANT_DATA_FOUND]" or is empty:
     * English: "I could not find that specific information in our database."
     * Korean: "해당 정보를 데이터베이스에서 찾을 수 없습니다."
   - STOP HERE. Do NOT add any other information.
   - Do NOT guess prices, fees, locations, or any specifics.
   - Do NOT use phrases like "typically", "usually", "generally", "based on".

2. HALLUCINATION PREVENTION:
   - If the question asks about something NOT in the Context, say "I don't have that information."
   - NEVER invent: prices (RM), dates, locations, names, or any factual claims.
   - If you're unsure, say "I couldn't find that" - NEVER guess.

3. CONTEXT-ONLY ANSWERING:
   - Answer ONLY using information explicitly stated in the Context.
   - If Context has partial data (e.g., GPA is present but Grades are missing), provide the available data (GPA) and state that the rest is unavailable.
   - DO NOT say "I could not find information" if you are providing some information (like GPA).
   - Quote specific values from Context when answering (e.g., "RM 94,195").

4. IDENTITY RULE:
   - If the user asks your name, answer "Buddy".
   - Service label is "UCSI buddy".
5. BE CONCISE: Maximum 3 sentences.
6. PLAIN TEXT ONLY: No markdown.
7. LANGUAGE: {lang_instruction}

Format: JSON ONLY
{{
   "text": "Your answer here...",
   "suggestions": ["Relevant Q1", "Relevant Q2", "Relevant Q3"]
}}
"""

    def _topic_hint(self, user_message: str) -> str:
        q = (user_message or "").lower()
        if "phd" in q or "doctoral" in q:
            return "PhD or doctoral programme details"
        if "scholarship" in q:
            return "scholarship details"
        if "cafeteria" in q or "dining" in q or "food" in q:
            return "cafeteria or dining details"
        if "address" in q or "location" in q:
            return "campus address details"
        if "print" in q:
            return "printing service details"
        if "prayer" in q:
            return "prayer room details"
        if "accommodation" in q or "hostel" in q:
            return "accommodation details"
        if "installment" in q:
            return "installment payment details"
        if "international" in q:
            return "international student details"
        return "that specific information"

    def _context_fallback_response(self, user_message: str, data_context: str) -> str:
        """
        Deterministic fallback when LLM call fails.
        Keeps response grounded in available context.
        """
        context = str(data_context or "")
        if not context or "[NO_RELEVANT_DATA_FOUND]" in context:
            return f"I cannot find {self._topic_hint(user_message)} in our database."

        snippet = context.split("\n\n")[0]
        snippet = re.sub(r"\[conf:[^\]]+\]", "", snippet, flags=re.IGNORECASE)
        snippet = snippet.replace("[Document]", "")
        snippet = re.sub(r"\s+", " ", snippet).strip()
        if not snippet:
            return f"I cannot find {self._topic_hint(user_message)} in our database."
        return snippet

    def _contains_token(self, text: str, keyword: str) -> bool:
        t = (text or "").lower()
        kw = (keyword or "").strip().lower()
        if not t or not kw:
            return False
        if " " in kw:
            return kw in t
        return re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", t) is not None

    def _is_university_domain_query(self, q: str) -> bool:
        domain_terms = [
            "ucsi", "campus", "hostel", "accommodation", "tuition", "fee", "fees",
            "scholarship", "programme", "program", "diploma", "degree", "phd",
            "faculty", "lecturer", "professor", "dean", "chancellor", "vice chancellor", "staff", "student", "intake",
            "semester", "block", "library", "shuttle", "gpa", "cgpa"
        ]
        return any(self._contains_token(q, term) for term in domain_terms)

    def _is_obvious_general_query(self, q: str) -> bool:
        if self._is_university_domain_query(q):
            return False
        patterns = [
            r"^what is\b",
            r"^who is\b",
            r"^who won\b",
            r"^where is the capital\b",
            r"^what is the currency\b",
            r"^what is the formula\b",
            r"^how to\b",
        ]
        if any(re.search(p, q) for p in patterns):
            return True
        generic_terms = [
            "python", "machine learning", "black hole", "einstein",
            "world cup", "capital of", "currency of", "h2o", "programming language"
        ]
        return any(term in q for term in generic_terms)

    def _safe_math_eval(self, expr: str):
        """Safely evaluate a basic arithmetic expression used in fallback answers."""
        if not expr:
            return None
        cleaned = str(expr).strip()
        if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", cleaned):
            return None

        try:
            node = ast.parse(cleaned, mode="eval")
        except Exception:
            return None

        allowed = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.UAdd, ast.USub
        )
        if any(not isinstance(n, allowed) for n in ast.walk(node)):
            return None

        try:
            value = eval(compile(node, "<expr>", "eval"), {"__builtins__": {}}, {})
        except Exception:
            return None

        if isinstance(value, (int, float)):
            if isinstance(value, float) and value.is_integer():
                return int(value)
            return value
        return None

    def _general_knowledge_fallback_response(self, user_message: str) -> str:
        """
        Deterministic fallback for obvious general knowledge questions.
        Used when LLM is rate-limited/busy to keep UX stable.
        """
        q = (user_message or "").strip().lower()
        if not q:
            return ""

        lookup = [
            (r"albert einstein", "Albert Einstein was a theoretical physicist best known for the theory of relativity."),
            (r"isaac newton|who is newton", "Isaac Newton was a physicist and mathematician known for the laws of motion and gravity."),
            (r"marie curie", "Marie Curie was a physicist and chemist known for pioneering research on radioactivity."),
            (r"mona lisa|who painted", "The Mona Lisa was painted by Leonardo da Vinci."),
            (r"capital of france", "The capital of France is Paris."),
            (r"capital of italy", "The capital of Italy is Rome."),
            (r"photosynthesis", "Photosynthesis is the process plants use to convert sunlight, water, and carbon dioxide into energy."),
            (r"python programming", "Python is a high-level programming language known for readable syntax and broad use in web, data, and AI."),
            (r"java programming|what is java", "Java is a general-purpose programming language widely used for enterprise, Android, and backend development."),
            (r"\bc programming\b|what is c language|what is c programming", "C is a low-level, high-performance programming language used for systems and embedded software."),
            (r"what is sql|\bsql\b", "SQL is a language used to query and manage data in relational databases."),
            (r"speed of light", "The speed of light in vacuum is about 299,792,458 meters per second."),
            (r"how many continents", "There are seven continents."),
            (r"largest ocean", "The Pacific Ocean is the largest ocean on Earth."),
            (r"romeo and juliet", "Romeo and Juliet was written by William Shakespeare."),
            (r"formula for water|h2o", "The chemical formula for water is H2O."),
            (r"tallest mountain", "Mount Everest is the tallest mountain above sea level."),
            (r"black hole", "A black hole is a region in space where gravity is so strong that even light cannot escape."),
            (r"quantum computing", "Quantum computing uses qubits and quantum effects, such as superposition and entanglement, to solve certain problems faster."),
            (r"machine learning", "Machine learning is a branch of AI where algorithms learn patterns from data to make predictions or decisions."),
            (r"what is recursion|explain recursion", "Recursion is a method where a function calls itself, with a base case to stop the process."),
            (r"what is gdp|\bgdp\b", "GDP stands for Gross Domestic Product, the total value of goods and services produced in an economy."),
            (r"currency of japan", "The currency of Japan is the Japanese yen."),
            (r"translate.*hello.*french|hello.*french", "In French, 'Hello' is 'Bonjour'."),
            (r"cook pasta", "Boil salted water, cook pasta until al dente, drain, then add sauce."),
            (r"world cup.*2022", "Argentina won the 2022 FIFA World Cup."),
            (r"meaning of life", "The meaning of life is a personal and philosophical question; many people frame it around purpose, relationships, and growth.")
        ]
        for pattern, answer in lookup:
            if re.search(pattern, q):
                return answer

        math_match = re.search(r"\bwhat is\s*([0-9\.\+\-\*\/\(\)\s]+)\??\s*$", q)
        if math_match:
            val = self._safe_math_eval(math_match.group(1))
            if val is not None:
                return f"The answer is {val}."

        if self._is_obvious_general_query(q):
            return "I can answer general questions, but the AI service is busy right now. Please try again shortly."
        return ""

    def _fast_intent_check(self, query: str) -> dict:
        """
        Heuristic-based fast intent check to bypass LLM latency.
        Returns None if checking requires LLM analysis.
        """
        q = query.lower().strip()
        
        # 1. Greetings (Fast Chit-Chat)
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "thanks", "thank you", "bye"]
        if q in greetings:
            return {
                "text": "Hello! How can I help you with UCSI University today?",
                "suggestions": ["Check my grades", "Hostel fees", "Bus schedule"],
                "needs_context": False,
                "search_term": None
            }

        # 2. Self Patterns (My Info)
        self_patterns = [
            "my grade", "my gpa", "my result", "my id", "who am i", "my schedule", "my profile", "my advisor",
            "where am i from", "where do i come from", "my nationality", "my country", "my major", "my programme", "my program"
        ]
        if any(p in q for p in self_patterns):
            return {"needs_context": True, "search_term": "self", "is_fast_path": True}

        # 2.5 General knowledge queries should stay out of RAG for better UX.
        if self._is_obvious_general_query(q):
            return {"force_general": True, "is_fast_path": True}

        # 3. University Keywords (Force RAG)
        uni_keywords = [
            "fee", "tuition", "cost", "price", 
            "hostel", "accommodation", "room", 
            "staff", "lecturer", "professor", "dean", "chancellor", "vice chancellor", "dr.", "dr ", "teacher",
            "course", "programme", "major", "diploma", "degree", "bachelor", "master", "phd",
            "facility", "library", "lab", "gym", "pool",
            "schedule", "class", "exam", "timetable", "bus", "shuttle"
        ]
        if any(self._contains_token(q, k) for k in uni_keywords):
            return {"needs_context": True, "search_term": None, "is_fast_path": True}
            
        return None # Fallback to LLM

    def process_message(self, user_message: str, data_context: str = "", conversation_history=None, language: str = "en") -> dict:
        """
        Unified processing with multi-language parsing
        Returns JSON: { "response": str, "suggestions": list, "needs_context": bool, "search_term": str }
        """
        if not self.client:
            return {"response": "System Error: AI Model not initialized.", "suggestions": []}

        # 0. FAST PATH INTENT CHECK (Prior to any LLM call)
        fast_force_general = False
        if not data_context:
            fast_result = self._fast_intent_check(user_message)
            if fast_result:
                print(f"[Fast Path] Intent Detected: {fast_result}")
                # If it's a direct response (like greeting), return immediately
                if "text" in fast_result:
                    return {
                        "response": fast_result["text"],
                        "suggestions": fast_result.get("suggestions", []),
                        "needs_context": False
                    }
                if fast_result.get("force_general"):
                    fallback = self._general_knowledge_fallback_response(user_message)
                    if fallback:
                        return {
                            "response": fallback,
                            "suggestions": ["Ask another general question", "Ask about UCSI information"],
                            "needs_context": False
                        }
                    fast_force_general = True
                else:
                # If it needs context, we return specific structure to trigger RAG in main.py
                # Note: main.py expects this structure from Phase 1
                    return {
                        "text": "Fetching information...",   # Placeholder, won't be shown if needs_context=True usually
                        "suggestions": [],
                        "needs_context": True,
                        "search_term": fast_result.get("search_term")
                    }

        try:
            # 1. Prepare Conversation Text
            conversation_text = ""
            if conversation_history:
                recent = conversation_history[-6:] # Keep it short to save tokens
                segments = []
                for item in recent:
                    role = "User" if item.get("role") == "user" else "Model"
                    content = item.get('content', '')
                    # Clean previous JSON outputs from history to avoid confusion
                    try:
                        c_json = json.loads(content)
                        if isinstance(c_json, dict):
                            content = c_json.get('text', '')
                    except:
                        pass
                    segments.append(f"{role}: {content}")
                conversation_text = "\n".join(segments)

            # 2. Construct Prompt (One-Shot Decision)
            lang_instruction = multilingual.get_ai_language_instruction(language)
            
            if data_context:
                # PHASE 2: We have data, generate answer.
                
                # CHECK FOR NEGATIVE LEARNING (BadQA)
                try:
                    from .db_engine import db_engine
                    bad_responses = db_engine.search_bad_responses(user_message)
                    if bad_responses:
                        warning = "\n\n[SYSTEM WARNING - LEARNED FROM PAST MISTAKES]:\n"
                        warning += "The following responses were marked as INCORRECT by users. DO NOT repeat these mistakes:\n"
                        for bad in bad_responses[:2]:  # Top 2 bad answers
                            warning += f"- BAD ANSWER: '{bad['bad_answer'][:100]}...'\n"
                            warning += f"  Reason: {bad['reason']}\n"
                        warning += "\nYou MUST provide a DIFFERENT, more accurate answer. If unsure, say 'I cannot find that information.'"
                        data_context += warning
                except Exception as e:
                    print(f"BadQA Check Error: {e}")

                # SELF-LEARNING PRIORITY (Positive)
                if "[Verified Answer]" in data_context:
                    data_context += "\n\n[SYSTEM NOTICE]: The context contains a 'Verified Answer'. This is a CONFIRMED CORRECT ANSWER from user feedback. Prioritize this information."


                prompt = self.qa_template.format(
                    context=data_context,
                    conversation=conversation_text,
                    question=user_message,
                    lang_instruction=lang_instruction
                )
            else:
                # PHASE 1: Intent Detection
                # STRICTER RULES FOR GENERAL UNI INFO
                if fast_force_general:
                    prompt = f"""You are UCSI buddy (name: Buddy), a concise assistant.

User Input: {user_message}

Instructions:
0. If the user asks your name, answer "Buddy".
1. Treat this as GENERAL KNOWLEDGE (not university database retrieval).
2. Answer directly using internal knowledge in at most 2 short sentences.
3. Plain text only. No markdown.
4. LANGUAGE: {lang_instruction}
5. JSON ONLY:
{{
  "text": "Direct answer",
  "suggestions": ["Related Q1", "Related Q2"],
  "needs_context": false,
  "search_term": null
}}
"""
                else:
                    prompt = f"""You are UCSI buddy (name: Buddy), a friendly university assistant. Analyze the user's input.

Current Conversation:
{conversation_text}

User Input: {user_message}

Instructions:
0. If the user asks your name, answer "Buddy" (service label: UCSI buddy).
1. Check Intent:
   - GREETING/CHIT-CHAT (hi, hello, thanks, who are you?): needs_context: false. Answer directly.
   - GENERAL KNOWLEDGE (Science, History, Math, Definitions, Coding, "What is X?", "Who is Y?", General Advice): needs_context: false. Answer strictly using internal knowledge.
     - SAFEGUARD: If you do not know the answer with 100% CERTAINTY, or if 'X' looks like a person's name who might be a student/staff (e.g. "Jeongbin", "Ali"), set needs_context: true to check the DB.
     - EXAMPLES (No Context): "What is Python?", "Who is Einstein?", "How to cook pasta?".
     - EXAMPLES (Need Context): "Who is Jeongbin?", "What is the Makers Lab?", "Tell me about Dr. Smith".
   - MY INFO/DATA (my grades, my id, who am i, my GPA): needs_context: true, search_term: "self".
   - PERSONAL & IMPLICIT (Where am I from? -> "nationality", Who is my advisor? -> "advisor", What is my major? -> "programme"): needs_context: true, search_term: extract specific keyword (e.g. "nationality").
   - STUDENT SEARCH (search for student X): needs_context: true, search_term: "name or id".
   - UNI DATA/FACTS (fees, courses, hostel, schedule, staff, facilities, "tell me about X"): needs_context: true. EVEN IF you think you know the answer, you MUST request context to ensure accuracy.
     - EXCEPTION: Only simple general concepts like "What is a GPA?" can be answered without context.
     - KEY: If the user asks for "University of ... fees" or "Diploma in ...", ALWAYS set needs_context: true.

2. SMART SUGGESTIONS: Generate 2-3 helpful follow-up questions that:
   - Are specific and actionable (not generic like "Tell me more")
   - Relate to the current topic or naturally extend the conversation
   - Examples for student info: "What is my current semester?", "Show my enrolled courses"
   - Examples for general: "What are the hostel prices?", "When is the registration deadline?"

3. PLAIN TEXT ONLY: No markdown like **bold** or *italics*. Write in plain sentences.
4. LANGUAGE: {lang_instruction}

5. Output Format (JSON ONLY):
   {{
      "text": "Response text (if not needing context)...", 
      "suggestions": ["Specific Q1", "Specific Q2", "Specific Q3"],
      "needs_context": true/false,
      "search_term": "keyword or null"
   }}
"""
            # 3. 🚀 ENTERPRISE-GRADE API CALL with Circuit Breaker, Rate Limiter, and Adaptive Backoff
            raw_text = None
            last_error = None
            
            # Check Circuit Breaker first
            if not _circuit_breaker.can_execute():
                cb_status = _circuit_breaker.get_status()
                if data_context:
                    return {
                        "response": self._context_fallback_response(user_message, data_context),
                        "suggestions": [],
                        "needs_context": False,
                        "error_type": "circuit_open_fallback",
                        "circuit_status": cb_status
                    }
                fallback = self._general_knowledge_fallback_response(user_message)
                if fallback:
                    return {
                        "response": fallback,
                        "suggestions": ["Ask another question", "Try again shortly"],
                        "needs_context": False,
                        "error_type": "circuit_open_general_fallback",
                        "circuit_status": cb_status
                    }
                return {
                    "response": f"AI service is temporarily paused for recovery. Please try again in {CIRCUIT_RECOVERY_TIMEOUT} seconds.",
                    "suggestions": ["Try again later", "Ask something simpler"],
                    "error_type": "circuit_open",
                    "circuit_status": cb_status
                }
            
            # Acquire rate limit token
            if not _rate_limiter.acquire(timeout=15.0):
                if data_context:
                    return {
                        "response": self._context_fallback_response(user_message, data_context),
                        "suggestions": [],
                        "needs_context": False,
                        "error_type": "rate_limited_fallback"
                    }
                fallback = self._general_knowledge_fallback_response(user_message)
                if fallback:
                    return {
                        "response": fallback,
                        "suggestions": ["Ask another question", "Try again shortly"],
                        "needs_context": False,
                        "error_type": "rate_limited_general_fallback"
                    }
                return {
                    "response": "The AI is currently handling many requests. Please wait a moment and try again.",
                    "suggestions": ["Wait and retry", "Ask another question"],
                    "error_type": "rate_limited"
                }
            
            for attempt in range(MAX_RETRIES):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt
                    )
                    raw_text = response.text.strip()
                    
                    # Record success to circuit breaker
                    _circuit_breaker.record_success()
                    break  # Success
                    
                except Exception as api_error:
                    last_error = api_error
                    error_str = str(api_error).lower()
                    
                    # Check if it's a rate limit error
                    if "resource_exhausted" in error_str or "quota" in error_str or "429" in error_str:
                        if attempt < MAX_RETRIES - 1:
                            # Adaptive Exponential Backoff with Jitter (Google SRE pattern)
                            base_wait = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                            jitter = random.uniform(0, JITTER_FACTOR * base_wait)
                            wait_time = base_wait + jitter
                            
                            print(f"[Rate Limit] Attempt {attempt + 1}/{MAX_RETRIES}. "
                                  f"Backing off {wait_time:.2f}s (base:{base_wait:.1f}s + jitter:{jitter:.2f}s)")
                            time.sleep(wait_time)
                            continue
                    else:
                        # Non-rate-limit error, record failure
                        _circuit_breaker.record_failure()
                    raise api_error
            
            if raw_text is None:
                _circuit_breaker.record_failure()
                raise last_error if last_error else Exception("API call failed")
            
            # 4. Parse JSON
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                # Normalize output keys
                return {
                    "response": data.get("text", "I'm thinking..."),
                    "suggestions": data.get("suggestions", []),
                    "needs_context": data.get("needs_context", False),
                    "search_term": data.get("search_term", None)
                }

            else:
                # Fallback for plain text response
                return {
                    "response": raw_text, 
                    "suggestions": ["Menu", "Contact"],
                    "needs_context": False
                }

        except Exception as e:
            error_msg = str(e).lower()
            print(f"AI Error: {e}")
            
            # Check for rate limit errors
            if "resource_exhausted" in error_msg or "quota" in error_msg or "429" in error_msg:
                if data_context:
                    return {
                        "response": self._context_fallback_response(user_message, data_context),
                        "suggestions": [],
                        "needs_context": False,
                        "error_type": "rate_limit_fallback"
                    }
                fallback = self._general_knowledge_fallback_response(user_message)
                if fallback:
                    return {
                        "response": fallback,
                        "suggestions": ["Ask another question", "Try again shortly"],
                        "needs_context": False,
                        "error_type": "rate_limit_general_fallback"
                    }
                return {
                    "response": "The AI service is temporarily busy. Please try again in a moment.",
                    "suggestions": ["Try again", "Ask something else"],
                    "error_type": "rate_limit"
                }

            if data_context:
                return {
                    "response": self._context_fallback_response(user_message, data_context),
                    "suggestions": [],
                    "needs_context": False,
                    "error_type": "generic_fallback"
                }

            fallback = self._general_knowledge_fallback_response(user_message)
            if fallback:
                return {
                    "response": fallback,
                    "suggestions": ["Ask another question", "Try again shortly"],
                    "needs_context": False,
                    "error_type": "general_fallback"
                }

            return {"response": "I'm having trouble processing your request. Please try again.", "suggestions": []}

    # ... (Unified process_message method kept above) ...
    # Deprecated fallback methods removed for cleanliness.
