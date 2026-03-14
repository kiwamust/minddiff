"""Token-based authentication."""

from fastapi import Cookie, HTTPException, status
from sqlalchemy.orm import Session

from minddiff.models import Member


def get_current_member(
    session: Session,
    token: str | None = Cookie(default=None, alias="minddiff_token"),
) -> Member:
    """Resolve member from cookie token."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    member = session.query(Member).filter(Member.token == token).first()
    if not member:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    return member
