"""Development seed data."""

from datetime import datetime, timedelta

from minddiff.db import create_tables, get_engine, get_session_factory
from minddiff.models import Team, Member, InputCycle, Response


def seed():
    engine = get_engine()
    create_tables(engine)
    Session = get_session_factory(engine)
    session = Session()

    # Team
    team = Team(name="Project Phoenix", cycle_interval=7)
    session.add(team)
    session.commit()
    session.refresh(team)
    print(f"Created team: {team.name} (id={team.id})")

    # Members
    members_data = [
        ("Sato", "sato@example.com", "pm"),
        ("Tanaka", "tanaka@example.com", "member"),
        ("Suzuki", "suzuki@example.com", "member"),
        ("Yamada", "yamada@example.com", "member"),
        ("Nakamura", "nakamura@example.com", "member"),
    ]
    members = []
    for name, email, role in members_data:
        m = Member(team_id=team.id, display_name=name, email=email, role=role)
        session.add(m)
        session.commit()
        session.refresh(m)
        members.append(m)
        print(f"  Member: {name} (token={m.token})")

    # Input Cycle
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
    session.refresh(cycle)
    print(f"Created cycle #{cycle.cycle_number} (id={cycle.id})")

    # Sample responses (3 members submitted, 2 pending)
    sample_responses = {
        members[0].id: {  # Sato (PM)
            1: "Q1リリースに向けた機能Xの実装完了が最優先。ユーザー獲得のためのMVPを3月末までに出す。",
            2: "バックエンドは順調。フロントが50%で遅れ気味。テストはまだ未着手。",
            3: "外部API連携の仕様変更リスク。スケジュールの遅延。チームの疲弊。",
            4: "APIスキーマはv2で確定。デプロイはCI/CD経由。木曜リリース。",
            5: "個人：ステークホルダーへの進捗報告。チーム：フロント実装の加速。",
        },
        members[1].id: {  # Tanaka
            1: "機能Xを動く状態にすること。品質は後から上げればいい。",
            2: "バックエンドOK。フロントはまあまあ進んでいる。テストは書き始めた。",
            3: "スケジュール。人手が足りない。",
            4: "APIスキーマv2確定。リリースは木曜。",
            5: "個人：APIエンドポイントの残り。チーム：テスト方針を決めること。",
        },
        members[2].id: {  # Suzuki
            1: "Q1リリースだが、本番品質まで仕上げる前提で動いている。MVPレベルでは不十分。",
            2: "バックエンド完了。フロントは設計から見直しが必要かもしれない。テスト未着手。",
            3: "フロントの設計が甘い。外部APIの仕様変更が最も心配。パフォーマンス要件が曖昧。",
            4: "APIスキーマv2は確定したはず。デプロイフローは未確認。",
            5: "個人：フロント設計のレビュー。チーム：品質基準の合意形成。",
        },
    }

    for member_id, dims in sample_responses.items():
        for dim, content in dims.items():
            resp = Response(
                input_cycle_id=cycle.id,
                member_id=member_id,
                dimension=dim,
                content=content,
                is_draft=False,
                submitted_at=now,
            )
            session.add(resp)
    session.commit()
    print(f"Created {sum(len(d) for d in sample_responses.values())} responses")

    session.close()
    print("\nSeed complete.")


if __name__ == "__main__":
    seed()
