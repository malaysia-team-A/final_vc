import datetime
import os
from functools import wraps
from hmac import compare_digest
from typing import Any, Dict, Optional

import jwt

try:
    from werkzeug.security import check_password_hash, generate_password_hash

    HAS_WERKZEUG = True
except Exception:
    HAS_WERKZEUG = False


SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("No SECRET_KEY set for application. Please set SECRET_KEY in .env.")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


def hash_password(password: str) -> str:
    """Hash a password for storage when werkzeug is available."""
    raw = str(password or "")
    if HAS_WERKZEUG:
        return generate_password_hash(raw)
    return raw


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain input against hashed/legacy plain-text stored values."""
    plain = str(plain_password or "")
    stored = str(hashed_password or "")
    if not plain or not stored:
        return False

    # Legacy plain-text support.
    if compare_digest(plain, stored):
        return True

    if HAS_WERKZEUG and stored.startswith(("pbkdf2:", "scrypt:")):
        try:
            return check_password_hash(stored, plain)
        except Exception:
            return False
    return False


def create_access_token(
    data: Dict[str, Any], expires_delta: Optional[datetime.timedelta] = None
) -> str:
    """Create a new JWT access token."""
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + (
        expires_delta or datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and verify a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def token_required(f):
    """Legacy compatibility wrapper.

    This project runs on FastAPI; new endpoints should use
    `app.api.dependencies.get_current_user` instead.
    """

    @wraps(f)
    def decorated(*args, **kwargs):
        raise RuntimeError(
            "token_required is deprecated. Use FastAPI dependency "
            "`get_current_user` from app.api.dependencies."
        )

    return decorated
