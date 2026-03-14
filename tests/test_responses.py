"""Test response submission and cycle management routes."""

from datetime import datetime, timedelta

from minddiff.models import Team, Member, InputCycle


def _setup_team_and_cycle(session):
    """Helper: create a team, member, and open cycle."""
    team = Team(name="Test Team", cycle_interval=7)
    session.add(team)
    session.commit()

    member = Member(team_id=team.id, display_name="Taro", email="taro@example.com", role="pm")
    session.add(member)
    session.commit()

    now = datetime.now()
    cycle = InputCycle(
        team_id=team.id, cycle_number=1, start_date=now, end_date=now + timedelta(days=7)
    )
    session.add(cycle)
    session.commit()

    session.refresh(team)
    session.refresh(member)
    session.refresh(cycle)
    return team, member, cycle


# --- Team API tests ---


def test_create_team(client):
    resp = client.post("/teams", json={"name": "New Team", "cycle_interval": 14})
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "New Team"
    assert data["cycle_interval"] == 14


def test_get_team(client, session):
    team = Team(name="Get Team")
    session.add(team)
    session.commit()
    session.refresh(team)

    resp = client.get(f"/teams/{team.id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Get Team"


def test_get_team_not_found(client):
    resp = client.get("/teams/9999")
    assert resp.status_code == 404


def test_add_member(client, session):
    team = Team(name="Member Team")
    session.add(team)
    session.commit()
    session.refresh(team)

    resp = client.post(
        f"/teams/{team.id}/members",
        json={"display_name": "Hanako", "email": "hanako@example.com", "role": "member"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["display_name"] == "Hanako"
    assert len(data["token"]) > 20


def test_list_members(client, session):
    team = Team(name="List Team")
    session.add(team)
    session.commit()
    session.refresh(team)

    for name in ["A", "B", "C"]:
        session.add(Member(team_id=team.id, display_name=name, email=f"{name}@example.com"))
    session.commit()

    resp = client.get(f"/teams/{team.id}/members")
    assert resp.status_code == 200
    assert len(resp.json()) == 3


def test_auth_by_token(client, session):
    team = Team(name="Auth Team")
    session.add(team)
    session.commit()

    member = Member(team_id=team.id, display_name="Auth User", email="auth@example.com")
    session.add(member)
    session.commit()
    session.refresh(member)

    resp = client.post(f"/teams/auth/{member.token}")
    assert resp.status_code == 200
    assert resp.json()["display_name"] == "Auth User"
    assert "minddiff_token" in resp.cookies


# --- Cycle API tests ---


def test_create_cycle(client, session):
    team = Team(name="Cycle Team")
    session.add(team)
    session.commit()
    session.refresh(team)

    resp = client.post("/cycles", json={"team_id": team.id})
    assert resp.status_code == 201
    data = resp.json()
    assert data["cycle_number"] == 1
    assert data["status"] == "open"


def test_create_second_cycle(client, session):
    team = Team(name="Multi Cycle")
    session.add(team)
    session.commit()
    session.refresh(team)

    client.post("/cycles", json={"team_id": team.id})
    resp = client.post("/cycles", json={"team_id": team.id})
    assert resp.json()["cycle_number"] == 2


def test_close_cycle(client, session):
    team = Team(name="Close Team")
    session.add(team)
    session.commit()
    session.refresh(team)

    resp = client.post("/cycles", json={"team_id": team.id})
    cycle_id = resp.json()["id"]

    resp = client.post(f"/cycles/{cycle_id}/close")
    assert resp.status_code == 200
    assert resp.json()["status"] == "closed"


# --- Response API tests ---


def test_save_and_update_response(client, session):
    team, member, cycle = _setup_team_and_cycle(session)

    # Save draft
    resp = client.put(
        f"/responses/{cycle.id}/{member.id}",
        json={"dimension": 1, "content": "Draft content", "is_draft": True},
    )
    assert resp.status_code == 200
    assert resp.json()["is_draft"] is True

    # Update to final
    resp = client.put(
        f"/responses/{cycle.id}/{member.id}",
        json={"dimension": 1, "content": "Final content", "is_draft": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"] == "Final content"
    assert data["is_draft"] is False
    assert data["submitted_at"] is not None


def test_response_invalid_dimension(client, session):
    team, member, cycle = _setup_team_and_cycle(session)

    resp = client.put(
        f"/responses/{cycle.id}/{member.id}",
        json={"dimension": 6, "content": "Bad", "is_draft": False},
    )
    assert resp.status_code == 400


def test_response_closed_cycle(client, session):
    team, member, cycle = _setup_team_and_cycle(session)
    cycle.status = "closed"
    session.commit()

    resp = client.put(
        f"/responses/{cycle.id}/{member.id}",
        json={"dimension": 1, "content": "Too late", "is_draft": False},
    )
    assert resp.status_code == 400


def test_get_member_responses(client, session):
    team, member, cycle = _setup_team_and_cycle(session)

    for dim in range(1, 4):
        client.put(
            f"/responses/{cycle.id}/{member.id}",
            json={"dimension": dim, "content": f"Dim {dim}", "is_draft": False},
        )

    resp = client.get(f"/responses/{cycle.id}/{member.id}")
    assert resp.status_code == 200
    assert len(resp.json()) == 3


def test_response_status(client, session):
    team, member, cycle = _setup_team_and_cycle(session)

    # Add another member
    m2 = Member(team_id=team.id, display_name="Jiro", email="jiro@example.com")
    session.add(m2)
    session.commit()
    session.refresh(m2)

    # Taro submits 3 dimensions
    for dim in range(1, 4):
        client.put(
            f"/responses/{cycle.id}/{member.id}",
            json={"dimension": dim, "content": f"Content {dim}", "is_draft": False},
        )

    resp = client.get(f"/responses/{cycle.id}/status")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2

    taro = next(d for d in data if d["display_name"] == "Taro")
    jiro = next(d for d in data if d["display_name"] == "Jiro")
    assert taro["submitted_dimensions"] == 3
    assert taro["complete"] is False
    assert jiro["submitted_dimensions"] == 0


# --- Report stub tests ---


def test_generate_report_stub(client, session):
    team, member, cycle = _setup_team_and_cycle(session)

    # Submit at least one response
    for dim in range(1, 6):
        client.put(
            f"/responses/{cycle.id}/{member.id}",
            json={"dimension": dim, "content": f"Content {dim}", "is_draft": False},
        )

    resp = client.post(f"/reports/{cycle.id}/generate")
    assert resp.status_code == 202
    assert resp.json()["status"] == "generated"


def test_generate_report_no_responses(client, session):
    team, member, cycle = _setup_team_and_cycle(session)

    resp = client.post(f"/reports/{cycle.id}/generate")
    assert resp.status_code == 400


def test_regenerate_report(client, session):
    """Regenerating a report should succeed (delete + recreate)."""
    team, member, cycle = _setup_team_and_cycle(session)

    client.put(
        f"/responses/{cycle.id}/{member.id}",
        json={"dimension": 1, "content": "Test", "is_draft": False},
    )
    resp1 = client.post(f"/reports/{cycle.id}/generate")
    assert resp1.status_code == 202

    resp2 = client.post(f"/reports/{cycle.id}/generate")
    assert resp2.status_code == 202


def test_get_report(client, session):
    team, member, cycle = _setup_team_and_cycle(session)

    client.put(
        f"/responses/{cycle.id}/{member.id}",
        json={"dimension": 1, "content": "Test", "is_draft": False},
    )
    client.post(f"/reports/{cycle.id}/generate")

    resp = client.get(f"/reports/{cycle.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert "synthesis" in data
    assert "divergences" in data
