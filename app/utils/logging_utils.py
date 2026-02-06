import logging
import re
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UCSI_Chatbot")

# Patterns to mask
PATTERNS = {
    "STUDENT_NUMBER": (r'\b\d{4,10}\b', '[STUDENT_ID]'),
    "NAME": (r'(?i)(name|student)["\']?\s*[:=]\s*["\']?([a-z\s]+)["\']?', r'\1: [NAME_REDACTED]'),
    "EMAIL": (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
    "PHONE": (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]')
}

def anonymize_text(text: str) -> str:
    """Mask PII in text"""
    if not isinstance(text, str):
        return str(text)
        
    for name, (pattern, replacement) in PATTERNS.items():
        text = re.sub(pattern, replacement, text)
    return text

def log_audit(action: str, user: str, details: str = ""):
    """Log an audit event with anonymization"""
    user_masked = anonymize_text(user)
    details_masked = anonymize_text(details)
    logger.info(f"AUDIT | Action: {action} | User: {user_masked} | Details: {details_masked}")
    
def get_logger():
    return logger
