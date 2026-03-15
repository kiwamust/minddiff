"""Test divergence detection service."""

import json
from pathlib import Path

from minddiff.services.divergence import (
    ALIGNMENT_SCORE_CAP,
    build_user_prompt,
    compute_alignment_score,
    detect_all_divergences,
    detect_divergence,
)
from minddiff.services.llm import LLMProvider


FIXTURES = Path(__file__).parent / "fixtures"


class MockDivergenceProvider(LLMProvider):
    """Mock LLM that returns predetermined divergence results with rubric."""

    def generate(self, system: str, user: str) -> str:
        return json.dumps(
            {
                "divergences": [
                    {
                        "concept": "「完成」の定義にズレがある",
                        "confidence": "高",
                        "evidence": [
                            "メンバー2: 「品質は後から上げればいい」",
                            "メンバー3: 「本番品質まで仕上げる前提」",
                        ],
                        "recommended_action": "Doneの定義を明示的に合意する",
                    }
                ],
                "rubric": {
                    "has_common_core": True,
                    "has_competing_goals": True,
                    "strong_evidence_count": 1,
                    "mention_agreement_ratio": "2/3",
                    "has_high_confidence_divergence": True,
                },
                "caution": "品質基準の認識差が潜在的なリスク",
            },
            ensure_ascii=False,
        )


class AllAgreeProvider(LLMProvider):
    """Mock LLM that returns a rubric indicating near-full agreement."""

    def generate(self, system: str, user: str) -> str:
        return json.dumps(
            {
                "divergences": [],
                "rubric": {
                    "has_common_core": True,
                    "has_competing_goals": False,
                    "strong_evidence_count": 0,
                    "mention_agreement_ratio": "3/3",
                    "has_high_confidence_divergence": False,
                },
                "caution": "表面的には一致",
            }
        )


class NoDivergenceProvider(LLMProvider):
    """Mock LLM that returns no divergences and no caution."""

    def generate(self, system: str, user: str) -> str:
        return json.dumps(
            {
                "divergences": [],
                "rubric": {
                    "has_common_core": True,
                    "has_competing_goals": False,
                    "strong_evidence_count": 0,
                    "mention_agreement_ratio": "3/3",
                    "has_high_confidence_divergence": False,
                },
            }
        )


def load_sample_responses():
    with open(FIXTURES / "sample_responses.json") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ── compute_alignment_score: basic tests ─────────────────────────────────


def test_compute_score_max():
    """Full agreement rubric produces ALIGNMENT_SCORE_CAP."""
    rubric = {
        "has_common_core": True,
        "has_competing_goals": False,
        "strong_evidence_count": 0,
        "mention_agreement_ratio": "3/3",
        "has_high_confidence_divergence": False,
    }
    assert compute_alignment_score(rubric) == ALIGNMENT_SCORE_CAP


def test_compute_score_competing_goals():
    rubric = {
        "has_common_core": True,
        "has_competing_goals": True,
        "strong_evidence_count": 0,
        "mention_agreement_ratio": "3/3",
        "has_high_confidence_divergence": False,
    }
    assert compute_alignment_score(rubric) == ALIGNMENT_SCORE_CAP - 0.20


def test_compute_score_all_penalties():
    """All penalties applied — minimum score."""
    rubric = {
        "has_common_core": False,
        "has_competing_goals": True,
        "strong_evidence_count": 3,
        "mention_agreement_ratio": "1/3",
        "has_high_confidence_divergence": True,
    }
    expected = 0.90 - 0.15 - 0.20 - 0.15 - 0.10 - 0.15
    assert compute_alignment_score(rubric) == expected


def test_compute_score_never_below_zero():
    rubric = {
        "has_common_core": False,
        "has_competing_goals": True,
        "strong_evidence_count": 10,
        "mention_agreement_ratio": "0/5",
        "has_high_confidence_divergence": True,
    }
    assert compute_alignment_score(rubric) >= 0.0


# ── compute_alignment_score: fail-closed tests ──────────────────────────


def test_compute_score_empty_rubric_fail_closed():
    """Empty rubric → all penalties applied (fail-closed, PRD 4.1)."""
    rubric = {}
    score = compute_alignment_score(rubric)
    assert score <= ALIGNMENT_SCORE_CAP
    # All penalties: no common core (-0.15), competing goals (-0.20),
    # high conf divergence (-0.15), malformed count (-0.10), no agreement (-0.15)
    expected = 0.90 - 0.15 - 0.20 - 0.15 - 0.10 - 0.15
    assert score == expected


def test_compute_score_malformed_ratio_fail_closed():
    """Malformed ratio → low agreement penalty (fail-closed, PRD 4.1)."""
    rubric = {
        "has_common_core": True,
        "has_competing_goals": False,
        "strong_evidence_count": 0,
        "mention_agreement_ratio": "invalid",
        "has_high_confidence_divergence": False,
    }
    score = compute_alignment_score(rubric)
    assert score == ALIGNMENT_SCORE_CAP - 0.15
    assert score < ALIGNMENT_SCORE_CAP


def test_compute_score_partial_rubric_fail_closed():
    """Partial rubric — missing fields get fail-closed defaults."""
    rubric = {"has_common_core": True}
    score = compute_alignment_score(rubric)
    # has_common_core=True (bool) → no penalty
    # has_competing_goals missing → fail_closed_default=True → -0.20
    # has_high_confidence_divergence missing → fail_closed_default=True → -0.15
    # strong_evidence_count missing → malformed → -0.10
    # mention_agreement_ratio missing → "" → -0.15
    expected = 0.90 - 0.20 - 0.15 - 0.10 - 0.15
    assert score == expected


# ── compute_alignment_score: strict bool enforcement ─────────────────────


def test_strict_bool_rejects_non_bool():
    """Non-bool types in boolean fields → fail-closed penalty."""
    rubric = {
        "has_common_core": "yes",  # str, not bool → fail_closed=False → -0.15
        "has_competing_goals": 0,  # int, not bool → fail_closed=True → -0.20
        "strong_evidence_count": 0,
        "mention_agreement_ratio": "3/3",
        "has_high_confidence_divergence": None,  # None, not bool → fail_closed=True → -0.15
    }
    score = compute_alignment_score(rubric)
    expected = 0.90 - 0.15 - 0.20 - 0.15
    assert score == expected


def test_strict_bool_accepts_real_bool():
    """True Python bools are accepted without penalty."""
    rubric = {
        "has_common_core": True,
        "has_competing_goals": False,
        "strong_evidence_count": 0,
        "mention_agreement_ratio": "3/3",
        "has_high_confidence_divergence": False,
    }
    assert compute_alignment_score(rubric) == ALIGNMENT_SCORE_CAP


def test_strict_bool_int_1_is_not_bool():
    """Python int 1 is NOT accepted as True — strict bool only."""
    rubric = {
        "has_common_core": 1,  # int, not bool → penalty
        "has_competing_goals": 0,  # int, not bool → penalty (fail_closed=True)
        "strong_evidence_count": 0,
        "mention_agreement_ratio": "3/3",
        "has_high_confidence_divergence": False,
    }
    score = compute_alignment_score(rubric)
    # has_common_core=1 (int) → fail_closed_default=False → -0.15
    # has_competing_goals=0 (int) → fail_closed_default=True → -0.20
    expected = 0.90 - 0.15 - 0.20
    assert score == expected


# ── compute_alignment_score: Dim5 two-layer tests ───────────────────────


def test_dim5_uses_personal_team_split():
    """Dim5 uses personal/team agreement ratios when available."""
    rubric = {
        "has_common_core": False,
        "has_competing_goals": True,
        "strong_evidence_count": 3,
        "mention_agreement_ratio": "2/3",  # should be overridden by split
        "has_high_confidence_divergence": True,
        "personal_agreement_ratio": "1/3",
        "team_agreement_ratio": "1/3",
    }
    score = compute_alignment_score(rubric, dimension=5)
    # has_common_core=False → -0.15
    # has_competing_goals=True → -0.20
    # has_high_conf=True → -0.15
    # strong_evidence_count=3 → -0.10
    # ratio = min(1/3, 1/3) = 0.33 < 0.5 → -0.15
    # misalignment: max(0.33, 0.33) < 0.5 → True → -0.05
    import pytest

    expected = 0.90 - 0.15 - 0.20 - 0.15 - 0.10 - 0.15 - 0.05
    assert score == pytest.approx(expected)


def test_dim5_takes_lower_ratio():
    """Dim5 uses min(personal, team) ratio — fail-closed."""
    rubric = {
        "has_common_core": True,
        "has_competing_goals": False,
        "strong_evidence_count": 0,
        "has_high_confidence_divergence": False,
        "personal_agreement_ratio": "3/3",  # high
        "team_agreement_ratio": "1/3",  # low → this wins
    }
    score = compute_alignment_score(rubric, dimension=5)
    # ratio = min(1.0, 0.33) = 0.33 < 0.5 → -0.15
    # misalignment: |1.0 - 0.33| = 0.67 > 0.34 → True → -0.05
    expected = 0.90 - 0.15 - 0.05
    assert score == expected


def test_dim5_fallback_to_mention_ratio():
    """Dim5 falls back to mention_agreement_ratio if split not provided."""
    rubric = {
        "has_common_core": True,
        "has_competing_goals": False,
        "strong_evidence_count": 0,
        "mention_agreement_ratio": "2/3",
        "has_high_confidence_divergence": False,
        # no personal/team split
    }
    score = compute_alignment_score(rubric, dimension=5)
    # ratio = 2/3 = 0.67, 0.5 ≤ 0.67 < 0.75 → -0.10
    # personal_team_misalignment missing → fail_closed=True → -0.05
    expected = 0.90 - 0.10 - 0.05
    assert score == expected


def test_dim5_both_high_ratios_no_misalignment():
    """Both ratios high and close → no misalignment penalty."""
    rubric = {
        "has_common_core": True,
        "has_competing_goals": False,
        "strong_evidence_count": 0,
        "has_high_confidence_divergence": False,
        "personal_agreement_ratio": "3/3",
        "team_agreement_ratio": "3/3",
    }
    score = compute_alignment_score(rubric, dimension=5)
    # misalignment: |1.0 - 1.0| = 0 ≤ 0.34, max = 1.0 ≥ 0.5 → False → no penalty
    assert score == ALIGNMENT_SCORE_CAP


def test_compute_misalignment_deterministic():
    """_compute_misalignment replaces LLM boolean judgment."""
    from minddiff.services.divergence import _compute_misalignment

    # Both low → misalignment
    assert _compute_misalignment(0.0, 0.0) is True
    assert _compute_misalignment(0.33, 0.33) is True

    # Ratios diverge → misalignment
    assert _compute_misalignment(1.0, 0.33) is True
    assert _compute_misalignment(0.33, 1.0) is True

    # Both high and close → no misalignment
    assert _compute_misalignment(1.0, 1.0) is False
    assert _compute_misalignment(0.67, 1.0) is False

    # Unparseable → fail-closed
    assert _compute_misalignment(-1.0, 1.0) is True
    assert _compute_misalignment(1.0, -1.0) is True
    assert _compute_misalignment(-1.0, -1.0) is True


def test_dim5_stance_aggregation():
    """Dim5 extract → aggregate: category/component/direction → deterministic fields."""
    from minddiff.services.divergence import (
        _compute_agreement_ratio,
        _compute_has_common_core_from_stances,
        _compute_has_competing_goals_from_stances,
    )

    # Ground truth fixture: all different
    stances_all_different = {
        "personal": [
            {
                "member": 1,
                "focus": "報告",
                "category": "communication",
                "component": "stakeholder",
                "direction": "alignment",
            },
            {
                "member": 2,
                "focus": "API実装",
                "category": "implementation",
                "component": "api",
                "direction": "speed",
            },
            {
                "member": 3,
                "focus": "設計レビュー",
                "category": "design",
                "component": "frontend",
                "direction": "quality",
            },
        ],
        "team": [
            {
                "member": 1,
                "focus": "フロント加速",
                "category": "delivery",
                "component": "frontend",
                "direction": "speed",
            },
            {
                "member": 2,
                "focus": "テスト方針",
                "category": "testing",
                "component": "testing",
                "direction": "alignment",
            },
            {
                "member": 3,
                "focus": "品質基準合意",
                "category": "quality",
                "component": "quality",
                "direction": "alignment",
            },
        ],
    }
    # Agreement: all categories distinct → 0/3
    assert _compute_agreement_ratio(["communication", "implementation", "design"]) == "0/3"
    assert _compute_agreement_ratio(["delivery", "testing", "quality"]) == "0/3"
    # Common core: no majority → False
    assert _compute_has_common_core_from_stances(stances_all_different) is False
    # Competing goals: frontend has speed + quality → True
    assert _compute_has_competing_goals_from_stances(stances_all_different) is True

    # Partial agreement: 2 of 3 in same category
    assert _compute_agreement_ratio(["quality", "testing", "quality"]) == "2/3"

    # Full agreement
    assert _compute_agreement_ratio(["delivery", "delivery", "delivery"]) == "3/3"

    # Common core: majority in one category → True
    stances_majority = {
        "personal": [
            {
                "member": 1,
                "focus": "A",
                "category": "implementation",
                "component": "api",
                "direction": "speed",
            },
            {
                "member": 2,
                "focus": "B",
                "category": "implementation",
                "component": "api",
                "direction": "speed",
            },
            {
                "member": 3,
                "focus": "C",
                "category": "design",
                "component": "frontend",
                "direction": "quality",
            },
        ],
        "team": [],
    }
    assert _compute_has_common_core_from_stances(stances_majority) is True

    # No competing goals: same component, same direction
    stances_aligned = {
        "personal": [
            {
                "member": 1,
                "focus": "A",
                "category": "delivery",
                "component": "frontend",
                "direction": "speed",
            },
        ],
        "team": [
            {
                "member": 2,
                "focus": "B",
                "category": "delivery",
                "component": "frontend",
                "direction": "speed",
            },
        ],
    }
    assert _compute_has_competing_goals_from_stances(stances_aligned) is False


def test_non_dim5_ignores_split_ratios():
    """Non-Dim5 dimensions ignore personal/team split fields."""
    rubric = {
        "has_common_core": True,
        "has_competing_goals": False,
        "strong_evidence_count": 0,
        "mention_agreement_ratio": "3/3",
        "has_high_confidence_divergence": False,
        "personal_agreement_ratio": "0/3",  # should be ignored for dim=1
        "team_agreement_ratio": "0/3",
        "personal_team_misalignment": True,
    }
    score = compute_alignment_score(rubric, dimension=1)
    assert score == ALIGNMENT_SCORE_CAP  # no penalties for dim 1


# ── Existing functional tests ────────────────────────────────────────────


def test_build_user_prompt():
    synthesis = {"summary": "Test synthesis", "common_themes": ["A"]}
    responses = [{"content": "回答1"}, {"content": "回答2"}]
    prompt = build_user_prompt(1, synthesis, responses)

    assert "目的理解" in prompt
    assert "Stage 1 統合結果" in prompt
    assert "メンバー1" in prompt
    assert "回答1" in prompt


def test_detect_divergence():
    provider = MockDivergenceProvider()
    synthesis = {"summary": "test"}
    responses = [{"content": "A"}, {"content": "B"}]

    result = detect_divergence(provider, 1, synthesis, responses)

    assert "divergences" in result
    assert len(result["divergences"]) == 1
    assert result["divergences"][0]["confidence"] == "高"
    assert isinstance(result["alignment_score"], float)
    assert result["alignment_score"] <= ALIGNMENT_SCORE_CAP


def test_alignment_score_cap():
    """PRD 4.1: alignment_score must never exceed ALIGNMENT_SCORE_CAP."""
    provider = AllAgreeProvider()
    result = detect_divergence(provider, 1, {}, [{"content": "test"}])

    assert result["alignment_score"] <= ALIGNMENT_SCORE_CAP
    assert result["alignment_score"] == ALIGNMENT_SCORE_CAP


def test_ensure_caution_when_no_divergence():
    """Each dimension must report at least one caution point."""
    provider = NoDivergenceProvider()
    result = detect_divergence(provider, 1, {}, [{"content": "test"}])

    assert result.get("caution") is not None
    assert len(result["caution"]) > 0


def test_detect_all_divergences():
    provider = MockDivergenceProvider()
    responses = load_sample_responses()
    synthesis = {str(dim): {"summary": "test"} for dim in range(1, 6)}

    divergences, alignment_scores = detect_all_divergences(provider, synthesis, responses)

    assert len(divergences) == 5
    assert len(alignment_scores) == 5
    for dim in range(1, 6):
        assert str(dim) in divergences
        assert str(dim) in alignment_scores
        assert alignment_scores[str(dim)] <= ALIGNMENT_SCORE_CAP


def test_detect_all_empty_responses():
    provider = MockDivergenceProvider()
    divergences, alignment_scores = detect_all_divergences(provider, {}, {})

    for dim in range(1, 6):
        assert divergences[str(dim)] == []
        assert alignment_scores[str(dim)] == 0.0
