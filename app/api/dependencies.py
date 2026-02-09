from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from app.utils.auth_utils import decode_access_token, verify_password
from app.engines.db_engine_async import db_engine_async

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")

async def get_db():
    return db_engine_async

async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload
