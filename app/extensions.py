from datetime import datetime
from collections import deque

# Global Engines (Initialized in main.py)
data_engine = None
ai_engine = None
feedback_engine = None
rag_engine = None
semantic_cache = None
faq_cache = None
unanswered_manager = None
query_rewriter = None
reranker = None

# In-memory storage
# Format: { "student_number": datetime_expiry }
high_security_sessions = {}

# In-memory conversation history
# Format: { session_key: deque }
CONVERSATION_HISTORY_LIMIT = 12
conversation_history_store = {}

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
