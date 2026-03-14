"""Pydantic schemas for request/response validation."""

from datetime import datetime

from pydantic import BaseModel


# --- Team ---


class TeamCreate(BaseModel):
    name: str
    cycle_interval: int = 7


class TeamOut(BaseModel):
    id: int
    name: str
    cycle_interval: int
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Member ---


class MemberCreate(BaseModel):
    display_name: str
    email: str
    role: str = "member"


class MemberOut(BaseModel):
    id: int
    team_id: int
    display_name: str
    email: str
    role: str
    created_at: datetime

    model_config = {"from_attributes": True}


class MemberCreateOut(MemberOut):
    """Returned only on member creation — includes the invite token."""

    token: str


# --- Auth ---


class AuthRequest(BaseModel):
    token: str


# --- Response ---


class ResponseSave(BaseModel):
    dimension: int
    content: str
    is_draft: bool = True


class ResponseOut(BaseModel):
    id: int
    input_cycle_id: int
    member_id: int
    dimension: int
    content: str
    is_draft: bool
    submitted_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Report ---


class ReportOut(BaseModel):
    id: int
    input_cycle_id: int
    synthesis: dict
    divergences: dict
    alignment_scores: dict
    generated_at: datetime

    model_config = {"from_attributes": True}
