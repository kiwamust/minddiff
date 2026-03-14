"""Team management routes."""

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from minddiff.dependencies import get_db
from minddiff.models import Team, Member
from minddiff.schemas import MemberCreate, MemberCreateOut, MemberOut, TeamCreate, TeamOut

router = APIRouter(prefix="/teams", tags=["teams"])


@router.post("", response_model=TeamOut, status_code=201)
def create_team(body: TeamCreate, db: Session = Depends(get_db)):
    team = Team(name=body.name, cycle_interval=body.cycle_interval)
    db.add(team)
    db.commit()
    db.refresh(team)
    return team


@router.get("/{team_id}", response_model=TeamOut)
def get_team(team_id: int, db: Session = Depends(get_db)):
    team = db.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return team


@router.post("/{team_id}/members", response_model=MemberCreateOut, status_code=201)
def add_member(team_id: int, body: MemberCreate, db: Session = Depends(get_db)):
    team = db.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    member = Member(
        team_id=team_id,
        display_name=body.display_name,
        email=body.email,
        role=body.role,
    )
    db.add(member)
    db.commit()
    db.refresh(member)
    return member


@router.get("/{team_id}/members", response_model=list[MemberOut])
def list_members(team_id: int, db: Session = Depends(get_db)):
    team = db.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return team.members


@router.post("/auth/{token}")
def auth_by_token(token: str, response: Response, db: Session = Depends(get_db)):
    """Authenticate via invite token, set cookie."""
    member = db.query(Member).filter(Member.token == token).first()
    if not member:
        raise HTTPException(status_code=404, detail="Invalid token")
    response.set_cookie(key="minddiff_token", value=token, httponly=True, samesite="lax")
    return {"display_name": member.display_name, "team_id": member.team_id, "role": member.role}
