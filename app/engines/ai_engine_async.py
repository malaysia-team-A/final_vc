import os
import re
import json
import asyncio
import random
import time
from datetime import datetime, timedelta
from google import genai
from .language_engine import multilingual
from .db_engine_async import db_engine_async

# ============================================================
# 🚀 ASYNC RESILIENCE CONFIGURATION
# ============================================================

def _env_int(name: str, default: int) -> int:
    try: return int(os.getenv(name, str(default)))
    except: return default

def _env_float(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)))
    except: return default

MAX_RETRIES = _env_int("AI_MAX_RETRIES", 2)
BASE_DELAY = _env_float("AI_BASE_DELAY", 0.5)
MAX_DELAY = _env_float("AI_MAX_DELAY", 2.0)
JITTER_FACTOR = _env_float("AI_JITTER_FACTOR", 0.2)

CIRCUIT_FAILURE_THRESHOLD = _env_int("AI_CIRCUIT_FAILURE_THRESHOLD", 8)
CIRCUIT_RECOVERY_TIMEOUT = _env_int("AI_CIRCUIT_RECOVERY_TIMEOUT", 8)
CIRCUIT_HALF_OPEN_MAX_CALLS = _env_int("AI_CIRCUIT_HALF_OPEN_MAX_CALLS", 3)

RATE_LIMIT_RPM = _env_int("AI_RATE_LIMIT_RPM", 120)
RATE_LIMIT_TOKENS = _env_int("AI_RATE_LIMIT_TOKENS", max(30, RATE_LIMIT_RPM))
RATE_LIMIT_REFILL_RATE = max(_env_float("AI_RATE_LIMIT_REFILL_RATE", RATE_LIMIT_RPM / 60.0), 0.1)

class AsyncCircuitBreaker:
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"
    
    def __init__(self):
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()
    
    async def can_execute(self) -> bool:
        async with self._lock:
            if self.state == self.CLOSED:
                return True
            elif self.state == self.OPEN:
                if self.last_failure_time and \
                   datetime.now() - self.last_failure_time > timedelta(seconds=CIRCUIT_RECOVERY_TIMEOUT):
                    self.state = self.HALF_OPEN
                    self.half_open_calls = 0
                    return True
                return False
            else: # HALF_OPEN
                if self.half_open_calls < CIRCUIT_HALF_OPEN_MAX_CALLS:
                    self.half_open_calls += 1
                    return True
                return False

    async def record_success(self):
        async with self._lock:
            self.state = self.CLOSED
            self.failure_count = 0

    async def record_failure(self):
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            if self.state == self.HALF_OPEN:
                self.state = self.OPEN
            elif self.failure_count >= CIRCUIT_FAILURE_THRESHOLD:
                self.state = self.OPEN

    async def get_status(self):
        async with self._lock:
            return {
                "state": self.state,
                "failure_count": self.failure_count,
                "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None
            }

class AsyncTokenBucketRateLimiter:
    def __init__(self):
        self.capacity = RATE_LIMIT_TOKENS
        self.tokens = RATE_LIMIT_TOKENS
        self.refill_rate = RATE_LIMIT_REFILL_RATE
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, timeout=10.0) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            async with self._lock:
                now = time.time()
                elapsed = now - self.last_refill
                self.tokens = min(self.capacity, self.tokens + (elapsed * self.refill_rate))
                self.last_refill = now
                
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
            await asyncio.sleep(0.1)
        return False

_circuit_breaker = AsyncCircuitBreaker()
_rate_limiter = AsyncTokenBucketRateLimiter()

class AsyncAIEngine:
    def __init__(self, model_name=None):
        self.raw_model_name = model_name or os.getenv("GEMINI_MODEL", "gemma-3-27b-it")
        # Prefer GEMINI_API_KEY, keep GOOGLE_API_KEY as legacy fallback.
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                # Initialize Async Client
                self.client = genai.Client(api_key=self.api_key, http_options={'api_version': 'v1beta'})
                self.model_name = self.raw_model_name.replace("models/", "")
                print(f"[AsyncAI] Initialized with model: {self.model_name}")
            except Exception as e:
                print(f"AsyncGemini Init Failed: {e}")
        else:
            print("[AsyncAI] Missing API key. Set GEMINI_API_KEY (or legacy GOOGLE_API_KEY).")
        
    def _topic_hint(self, user_message: str) -> str:
        q = (user_message or "").lower()
        if "scholarship" in q: return "scholarship details"
        return "that specific information"

    def _context_fallback_response(self, user_message: str, data_context: str) -> str:
        if not data_context or "[NO_RELEVANT_DATA_FOUND]" in data_context:
            return f"I cannot find {self._topic_hint(user_message)} in our database."
        snippet = data_context.split("\n\n")[0]
        snippet = re.sub(r"\[conf:[^\]]+\]", "", snippet)
        snippet = snippet.replace("[Document]", "").strip()
        return snippet if snippet else f"I cannot find {self._topic_hint(user_message)}."

    def _should_use_fast_path(self, query: str) -> dict:
        q = query.lower().strip()
        if q in ["hi", "hello", "hey"]:
            return {"text": "Hello! How can I help you?", "needs_context": False}
        return None

    def _sanitize_search_term(self, value) -> str | None:
        if value is None:
            return None
        token = str(value).strip()
        if not token:
            return None
        if len(token) > 64:
            token = token[:64]
        return token

    async def process_message(self, user_message: str, data_context: str = "", conversation_history=None, language: str = "en") -> dict:
        if not self.client:
            return {"response": "System Error: AI Model not initialized."}
            
        # Fast Path
        fast = self._should_use_fast_path(user_message)
        if fast: return {"response": fast["text"], "needs_context": False}
        
        # Prepare Context & History
        # (Same logic as sync version, omitted for brevity but should be included)
        
        # Construct Prompt
        lang_instruct = multilingual.get_ai_language_instruction(language)
        prompt = f"""You are UCSI Buddy.
Context: {data_context}
Question: {user_message}
Output JSON: {{ "text": "...", "suggestions": [], "needs_context": bool, "search_term": string|null }}
Rules:
1. If Context has [NO_RELEVANT_DATA_FOUND], say "I cannot find that info".
2. Language: {lang_instruct}
3. If question is about UCSI/campus/hostel/fees/programme/staff/schedule/personal student data,
   set needs_context=true and provide short search_term.
4. If question is generic world knowledge, set needs_context=false.
"""

        # Resilience Checks
        if not await _circuit_breaker.can_execute():
             return {"response": "Service paused for recovery.", "error_type": "circuit_open"}
        
        if not await _rate_limiter.acquire(timeout=10.0):
             return {"response": "Service busy, please wait.", "error_type": "rate_limited"}

        # API Call
        try:
            for attempt in range(MAX_RETRIES):
                try:
                    # Async Generation
                    response = await self.client.aio.models.generate_content(
                        model=self.model_name,
                        contents=prompt
                    )
                    text = response.text
                    await _circuit_breaker.record_success()
                    
                    # Parse JSON
                    try:
                        # extract json
                        match = re.search(r'\{.*\}', text, re.DOTALL)
                        if match:
                            data = json.loads(match.group())
                            suggestions = data.get("suggestions")
                            if not isinstance(suggestions, list):
                                suggestions = []
                            return {
                                "response": data.get("text") or text,
                                "suggestions": suggestions[:3],
                                "needs_context": bool(data.get("needs_context", False)),
                                "search_term": self._sanitize_search_term(data.get("search_term")),
                            }
                    except: pass
                    
                    return {"response": text, "needs_context": False, "search_term": None}
                    
                except Exception as e:
                    if "resource_exhausted" in str(e).lower():
                        await asyncio.sleep(BASE_DELAY * (2 ** attempt))
                        continue
                    await _circuit_breaker.record_failure()
                    raise e
                    
        except Exception as e:
            print(f"Async AI Error: {e}")
            return {"response": "I encountered an error processing your request.", "error_type": "ai_error"}

# Singleton
ai_engine_async = AsyncAIEngine()
