"""Test ORM models and DB constraints."""

from datetime import datetime, timedelta

from minddiff.models import Team, Member, InputCycle, Response, Report


def test_create_team(session):
    team = Team(name="Alpha Team", cycle_interval=7)
    session.add(team)
    session.commit()

    assert team.id is not None
    assert team.name == "Alpha Team"
    assert team.cycle_interval == 7


def test_create_member_with_token(session):
    team = Team(name="Test Team")
    session.add(team)
    session.commit()

    member = Member(team_id=team.id, display_name="Taro", email="taro@example.com", role="pm")
    session.add(member)
    session.commit()

    assert member.token is not None
    assert len(member.token) > 20
    assert member.role == "pm"


def test_member_team_relationship(session):
    team = Team(name="Rel Team")
    session.add(team)
    session.commit()

    m1 = Member(team_id=team.id, display_name="A", email="a@example.com")
    m2 = Member(team_id=team.id, display_name="B", email="b@example.com")
    session.add_all([m1, m2])
    session.commit()

    session.refresh(team)
    assert len(team.members) == 2


def test_create_input_cycle(session):
    team = Team(name="Cycle Team")
    session.add(team)
    session.commit()

    now = datetime.now()
    cycle = InputCycle(
        team_id=team.id,
        cycle_number=1,
        start_date=now,
        end_date=now + timedelta(days=7),
        status="open",
    )
    session.add(cycle)
    session.commit()

    assert cycle.id is not None
    assert cycle.status == "open"


def test_create_response(session):
    team = Team(name="Resp Team")
    session.add(team)
    session.commit()

    member = Member(team_id=team.id, display_name="C", email="c@example.com")
    session.add(member)
    session.commit()

    now = datetime.now()
    cycle = InputCycle(
        team_id=team.id, cycle_number=1, start_date=now, end_date=now + timedelta(days=7)
    )
    session.add(cycle)
    session.commit()

    resp = Response(
        input_cycle_id=cycle.id,
        member_id=member.id,
        dimension=1,
        content="Q1 release is the top priority",
        is_draft=False,
        submitted_at=now,
    )
    session.add(resp)
    session.commit()

    assert resp.id is not None
    assert resp.dimension == 1
    assert resp.is_draft is False


def test_create_report_with_json(session):
    team = Team(name="Report Team")
    session.add(team)
    session.commit()

    now = datetime.now()
    cycle = InputCycle(
        team_id=team.id, cycle_number=1, start_date=now, end_date=now + timedelta(days=7)
    )
    session.add(cycle)
    session.commit()

    import json

    report = Report(
        input_cycle_id=cycle.id,
        synthesis=json.dumps({"dim_1": "Everyone agrees on Q1 release"}),
        divergences=json.dumps({"dim_1": [{"concept": "Definition of Done", "confidence": "high"}]}),
        alignment_scores=json.dumps({"dim_1": 0.65, "dim_2": 0.80}),
    )
    session.add(report)
    session.commit()

    assert report.get_synthesis()["dim_1"] == "Everyone agrees on Q1 release"
    assert report.get_alignment_scores()["dim_1"] == 0.65
    assert report.get_divergences()["dim_1"][0]["confidence"] == "high"


def test_cascade_delete_team(session):
    team = Team(name="Cascade Team")
    session.add(team)
    session.commit()

    member = Member(team_id=team.id, display_name="D", email="d@example.com")
    session.add(member)
    session.commit()

    session.delete(team)
    session.commit()

    assert session.query(Member).count() == 0


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
