from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
from typing import List, Optional

class LoginRequest(BaseModel):
    student_number: str
    username: Optional[str] = None
    name: Optional[str] = None

    model_config = ConfigDict(extra="ignore")

class VerifyPasswordRequest(BaseModel):
    student_number: Optional[str] = None
    password: str

    model_config = ConfigDict(extra="ignore")

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    student_name: str

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = Field(default=None, alias="session_id")
    search_term: Optional[str] = None
    needs_context: bool = False

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

class ChatResponse(BaseModel):
    text: str
    suggestions: List[str] = []
    conversation_id: str
    error: Optional[str] = None

class FeedbackRequest(BaseModel):
    user_message: str = Field(validation_alias=AliasChoices("user_message", "question"))
    ai_response: str = Field(validation_alias=AliasChoices("ai_response", "answer"))
    rating: str  # 'positive' or 'negative'
    comment: Optional[str] = None
    session_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("session_id", "conversation_id"),
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    @field_validator("rating")
    @classmethod
    def _normalize_rating(cls, value: str) -> str:
        rating = str(value or "").strip().lower()
        if rating not in {"positive", "negative"}:
            raise ValueError("rating must be 'positive' or 'negative'")
        return rating
