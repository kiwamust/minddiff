"""FastAPI dependency injection."""

from typing import Generator

from fastapi import Cookie, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from minddiff.models import InputCycle, Member


def get_db(request: Request) -> Generator[Session, None, None]:
    """Yield a DB session per request."""
    session_factory = request.app.state.session_factory
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def get_current_member(
    db: Session = Depends(get_db),
    minddiff_token: str | None = Cookie(default=None),
) -> Member:
    """Resolve member from cookie token. Raises 401 if invalid."""
    if not minddiff_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    member = db.query(Member).filter(Member.token == minddiff_token).first()
    if not member:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return member


def require_team_access(member: Member, cycle: InputCycle):
    """Ensure member belongs to the cycle's team."""
    if member.team_id != cycle.team_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")


def require_pm(member: Member):
    """Ensure member has PM role."""
    if member.role != "pm":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="PM role required")
