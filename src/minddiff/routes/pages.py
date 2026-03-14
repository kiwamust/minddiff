"""HTML page routes (htmx/Jinja2)."""

from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from minddiff.dependencies import get_db
from minddiff.models import Member, InputCycle, Response, Report

router = APIRouter(tags=["pages"])

DIMENSIONS = {
    1: "目的理解",
    2: "状況認識",
    3: "リスク認知",
    4: "合意事項",
    5: "優先順位",
}


def _get_member_from_cookie(request: Request, db: Session) -> Member | None:
    token = request.cookies.get("minddiff_token")
    if not token:
        return None
    return db.query(Member).filter(Member.token == token).first()


@router.get("/", response_class=HTMLResponse)
def index(request: Request, db: Session = Depends(get_db)):
    templates = request.app.state.templates
    member = _get_member_from_cookie(request, db)
    if not member:
        return RedirectResponse("/login", status_code=302)

    team = member.team
    cycles = (
        db.query(InputCycle)
        .filter(InputCycle.team_id == team.id)
        .order_by(InputCycle.cycle_number.desc())
        .all()
    )
    current_cycle = cycles[0] if cycles else None
    report = None
    if current_cycle:
        report = db.query(Report).filter(Report.input_cycle_id == current_cycle.id).first()

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "member": member,
            "team": team,
            "cycles": cycles,
            "current_cycle": current_cycle,
            "report": report,
        },
    )


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@router.post("/login")
def login(request: Request, token: str = Form(...), db: Session = Depends(get_db)):
    member = db.query(Member).filter(Member.token == token).first()
    if not member:
        templates = request.app.state.templates
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "無効なトークンです"},
            status_code=401,
        )
    response = RedirectResponse("/", status_code=302)
    response.set_cookie(key="minddiff_token", value=token, httponly=True, samesite="lax")
    return response


@router.get("/input/{cycle_id}", response_class=HTMLResponse)
def input_page(request: Request, cycle_id: int, db: Session = Depends(get_db)):
    templates = request.app.state.templates
    member = _get_member_from_cookie(request, db)
    if not member:
        return RedirectResponse("/login", status_code=302)

    cycle = db.get(InputCycle, cycle_id)
    if not cycle:
        raise HTTPException(status_code=404, detail="Cycle not found")

    # Current responses for this cycle
    responses = (
        db.query(Response)
        .filter(Response.input_cycle_id == cycle_id, Response.member_id == member.id)
        .all()
    )
    current_responses = {r.dimension: r.content for r in responses}

    # Previous cycle responses for reference (FR-03)
    previous_responses = {}
    prev_cycle = (
        db.query(InputCycle)
        .filter(InputCycle.team_id == cycle.team_id, InputCycle.cycle_number < cycle.cycle_number)
        .order_by(InputCycle.cycle_number.desc())
        .first()
    )
    if prev_cycle:
        prev_resps = (
            db.query(Response)
            .filter(Response.input_cycle_id == prev_cycle.id, Response.member_id == member.id)
            .all()
        )
        previous_responses = {r.dimension: r.content for r in prev_resps}

    return templates.TemplateResponse(
        "input.html",
        {
            "request": request,
            "member": member,
            "cycle": cycle,
            "current_responses": current_responses,
            "previous_responses": previous_responses,
        },
    )


@router.get("/report/{cycle_id}", response_class=HTMLResponse)
def report_page(request: Request, cycle_id: int, db: Session = Depends(get_db)):
    templates = request.app.state.templates
    member = _get_member_from_cookie(request, db)
    if not member:
        return RedirectResponse("/login", status_code=302)

    cycle = db.get(InputCycle, cycle_id)
    if not cycle:
        raise HTTPException(status_code=404, detail="Cycle not found")

    report = db.query(Report).filter(Report.input_cycle_id == cycle_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    return templates.TemplateResponse(
        "report.html",
        {
            "request": request,
            "member": member,
            "cycle": cycle,
            "report": report,
            "synthesis": report.get_synthesis(),
            "divergences": report.get_divergences(),
            "alignment_scores": report.get_alignment_scores(),
        },
    )


@router.post("/new-cycle")
def create_cycle_page(request: Request, team_id: int = Form(...), db: Session = Depends(get_db)):
    """Create cycle from dashboard form (PM only)."""
    from datetime import datetime, timedelta
    from minddiff.models import Team

    team = db.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404)

    last_cycle = (
        db.query(InputCycle)
        .filter(InputCycle.team_id == team_id)
        .order_by(InputCycle.cycle_number.desc())
        .first()
    )
    next_number = (last_cycle.cycle_number + 1) if last_cycle else 1
    now = datetime.now()

    cycle = InputCycle(
        team_id=team_id,
        cycle_number=next_number,
        start_date=now,
        end_date=now + timedelta(days=team.cycle_interval),
        status="open",
    )
    db.add(cycle)
    db.commit()
    return RedirectResponse("/", status_code=302)


@router.get("/partials/status/{cycle_id}", response_class=HTMLResponse)
def status_partial(request: Request, cycle_id: int, db: Session = Depends(get_db)):
    """Return HTML partial for member completion status (htmx)."""
    templates = request.app.state.templates
    cycle = db.get(InputCycle, cycle_id)
    if not cycle:
        return HTMLResponse("")

    members = db.query(Member).filter(Member.team_id == cycle.team_id).all()
    statuses = []
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
        statuses.append(
            {
                "display_name": m.display_name,
                "submitted": submitted,
                "complete": submitted == 5,
            }
        )

    return templates.TemplateResponse(
        "partials/status_panel.html",
        {"request": request, "statuses": statuses},
    )
