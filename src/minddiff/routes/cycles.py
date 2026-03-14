"""Input cycle management routes."""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from minddiff.dependencies import get_db
from minddiff.models import Team, InputCycle

router = APIRouter(prefix="/cycles", tags=["cycles"])


class CycleCreate(BaseModel):
    team_id: int


class CycleOut(BaseModel):
    id: int
    team_id: int
    cycle_number: int
    start_date: datetime
    end_date: datetime
    status: str

    model_config = {"from_attributes": True}


@router.post("", response_model=CycleOut, status_code=201)
def create_cycle(body: CycleCreate, db: Session = Depends(get_db)):
    team = db.get(Team, body.team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    last_cycle = (
        db.query(InputCycle)
        .filter(InputCycle.team_id == body.team_id)
        .order_by(InputCycle.cycle_number.desc())
        .first()
    )
    next_number = (last_cycle.cycle_number + 1) if last_cycle else 1
    now = datetime.now()

    cycle = InputCycle(
        team_id=body.team_id,
        cycle_number=next_number,
        start_date=now,
        end_date=now + timedelta(days=team.cycle_interval),
        status="open",
    )
    db.add(cycle)
    db.commit()
    db.refresh(cycle)
    return cycle


@router.get("/{cycle_id}", response_model=CycleOut)
def get_cycle(cycle_id: int, db: Session = Depends(get_db)):
    cycle = db.get(InputCycle, cycle_id)
    if not cycle:
        raise HTTPException(status_code=404, detail="Cycle not found")
    return cycle


@router.get("", response_model=list[CycleOut])
def list_cycles(team_id: int, db: Session = Depends(get_db)):
    return (
        db.query(InputCycle)
        .filter(InputCycle.team_id == team_id)
        .order_by(InputCycle.cycle_number.desc())
        .all()
    )


@router.post("/{cycle_id}/close", response_model=CycleOut)
def close_cycle(cycle_id: int, db: Session = Depends(get_db)):
    cycle = db.get(InputCycle, cycle_id)
    if not cycle:
        raise HTTPException(status_code=404, detail="Cycle not found")
    cycle.status = "closed"
    db.commit()
    db.refresh(cycle)
    return cycle
