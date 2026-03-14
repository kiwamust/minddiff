"""Report viewing and generation routes."""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from minddiff.dependencies import get_db
from minddiff.models import InputCycle, Report, Response
from minddiff.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/{cycle_id}")
def get_report(cycle_id: int, db: Session = Depends(get_db)):
    """Get the report for a cycle."""
    report = db.query(Report).filter(Report.input_cycle_id == cycle_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return {
        "id": report.id,
        "input_cycle_id": report.input_cycle_id,
        "synthesis": report.get_synthesis(),
        "divergences": report.get_divergences(),
        "alignment_scores": report.get_alignment_scores(),
        "generated_at": report.generated_at,
    }


def _collect_responses(cycle_id: int, db: Session) -> dict[int, list[dict]]:
    """Collect submitted responses grouped by dimension."""
    responses = (
        db.query(Response)
        .filter(Response.input_cycle_id == cycle_id, Response.is_draft == False)  # noqa: E712
        .order_by(Response.dimension, Response.member_id)
        .all()
    )
    by_dimension: dict[int, list[dict]] = {}
    for r in responses:
        by_dimension.setdefault(r.dimension, []).append(
            {
                "member_id": r.member_id,
                "content": r.content,
            }
        )
    return by_dimension


@router.post("/{cycle_id}/generate", status_code=202)
def trigger_report_generation(cycle_id: int, db: Session = Depends(get_db)):
    """Generate report using LLM (or stub if no API key)."""
    cycle = db.get(InputCycle, cycle_id)
    if not cycle:
        raise HTTPException(status_code=404, detail="Cycle not found")

    by_dimension = _collect_responses(cycle_id, db)
    submitted_member_count = len({r["member_id"] for resps in by_dimension.values() for r in resps})
    if submitted_member_count == 0:
        raise HTTPException(status_code=400, detail="No submitted responses yet")

    # Delete existing report if regenerating
    existing = db.query(Report).filter(Report.input_cycle_id == cycle_id).first()
    if existing:
        db.delete(existing)
        db.commit()

    # Try LLM generation, fall back to stub
    if settings.anthropic_api_key:
        try:
            synthesis, divergences, alignment_scores, raw = _generate_with_llm(by_dimension)
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            synthesis, divergences, alignment_scores, raw = _generate_stub(by_dimension)
    else:
        synthesis, divergences, alignment_scores, raw = _generate_stub(by_dimension)

    report = Report(
        input_cycle_id=cycle_id,
        synthesis=json.dumps(synthesis, ensure_ascii=False),
        divergences=json.dumps(divergences, ensure_ascii=False),
        alignment_scores=json.dumps(alignment_scores, ensure_ascii=False),
        raw_llm_response=raw,
    )
    db.add(report)
    cycle.status = "reported"
    db.commit()
    db.refresh(report)
    return {"report_id": report.id, "status": "generated"}


def _generate_with_llm(by_dimension: dict[int, list[dict]]) -> tuple[dict, dict, dict, str]:
    """Generate report using Claude API."""
    from minddiff.services.llm import ClaudeProvider
    from minddiff.services.synthesis import synthesize_all
    from minddiff.services.divergence import detect_all_divergences

    provider = ClaudeProvider(api_key=settings.anthropic_api_key)

    synthesis = synthesize_all(provider, by_dimension)
    divergences, alignment_scores = detect_all_divergences(provider, synthesis, by_dimension)

    raw = json.dumps(
        {"synthesis": synthesis, "divergences": divergences, "alignment_scores": alignment_scores},
        ensure_ascii=False,
    )
    return synthesis, divergences, alignment_scores, raw


def _generate_stub(by_dimension: dict[int, list[dict]]) -> tuple[dict, dict, dict, str]:
    """Generate stub report without LLM."""
    synthesis = {}
    divergences = {}
    alignment_scores = {}

    for dim in range(1, 6):
        dim_key = str(dim)
        resps = by_dimension.get(dim, [])
        member_count = len(resps)

        if member_count > 0:
            contents = [r["content"] for r in resps]
            synthesis[dim_key] = {
                "summary": f"{member_count}名の回答を受領。LLM統合はAPI key設定後に実行可能。",
                "common_themes": [],
                "mention_distribution": {},
            }
            divergences[dim_key] = [
                {
                    "concept": "自動分析未実行",
                    "confidence": "低",
                    "evidence": [f"メンバー{i+1}: 「{c[:50]}...」" for i, c in enumerate(contents)],
                    "recommended_action": "LLM統合を有効にしてレポートを再生成してください",
                }
            ]
            alignment_scores[dim_key] = 0.5
        else:
            synthesis[dim_key] = {"summary": "回答なし"}
            divergences[dim_key] = []
            alignment_scores[dim_key] = 0.0

    return synthesis, divergences, alignment_scores, "stub"
