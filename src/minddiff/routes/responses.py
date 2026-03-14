"""Response submission routes."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from minddiff.dependencies import get_db
from minddiff.models import InputCycle, Member, Response
from minddiff.schemas import ResponseSave, ResponseOut

router = APIRouter(prefix="/responses", tags=["responses"])


# Static path MUST come before parameterized /{cycle_id}/{member_id}
@router.get("/{cycle_id}/status")
def response_status(cycle_id: int, db: Session = Depends(get_db)):
    """Get response completion status per member for a cycle."""
    cycle = db.get(InputCycle, cycle_id)
    if not cycle:
        raise HTTPException(status_code=404, detail="Cycle not found")

    members = db.query(Member).filter(Member.team_id == cycle.team_id).all()
    result = []
    for m in members:
        submitted = (
            db.query(Response)
            .filter(
                Response.input_cycle_id == cycle_id,
                Response.member_id == m.id,
                Response.is_draft == False,  # noqa: E712
            )
            .count()
        )
        result.append(
            {
                "member_id": m.id,
                "display_name": m.display_name,
                "submitted_dimensions": submitted,
                "total_dimensions": 5,
                "complete": submitted == 5,
            }
        )
    return result


@router.put("/{cycle_id}/{member_id}", response_model=ResponseOut)
def save_response(
    cycle_id: int,
    member_id: int,
    body: ResponseSave,
    db: Session = Depends(get_db),
):
    """Save or update a response (draft or final)."""
    cycle = db.get(InputCycle, cycle_id)
    if not cycle:
        raise HTTPException(status_code=404, detail="Cycle not found")
    if cycle.status != "open":
        raise HTTPException(status_code=400, detail="Cycle is not open")

    member = db.get(Member, member_id)
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")

    if body.dimension < 1 or body.dimension > 5:
        raise HTTPException(status_code=400, detail="Dimension must be 1-5")

    existing = (
        db.query(Response)
        .filter(
            Response.input_cycle_id == cycle_id,
            Response.member_id == member_id,
            Response.dimension == body.dimension,
        )
        .first()
    )

    now = datetime.now()
    if existing:
        existing.content = body.content
        existing.is_draft = body.is_draft
        existing.updated_at = now
        if not body.is_draft:
            existing.submitted_at = now
        db.commit()
        db.refresh(existing)
        return existing

    resp = Response(
        input_cycle_id=cycle_id,
        member_id=member_id,
        dimension=body.dimension,
        content=body.content,
        is_draft=body.is_draft,
        submitted_at=None if body.is_draft else now,
    )
    db.add(resp)
    db.commit()
    db.refresh(resp)
    return resp


@router.get("/{cycle_id}/{member_id}", response_model=list[ResponseOut])
def get_member_responses(cycle_id: int, member_id: int, db: Session = Depends(get_db)):
    """Get all responses for a member in a cycle."""
    return (
        db.query(Response)
        .filter(Response.input_cycle_id == cycle_id, Response.member_id == member_id)
        .order_by(Response.dimension)
        .all()
    )
