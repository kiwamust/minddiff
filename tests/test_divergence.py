"""Test divergence detection service."""

import json
from pathlib import Path

from minddiff.services.divergence import (
    build_user_prompt,
    detect_divergence,
    detect_all_divergences,
    ALIGNMENT_SCORE_CAP,
)
from minddiff.services.llm import LLMProvider


FIXTURES = Path(__file__).parent / "fixtures"


class MockDivergenceProvider(LLMProvider):
    """Mock LLM that returns predetermined divergence results."""

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
                "alignment_score": 0.55,
                "caution": "品質基準の認識差が潜在的なリスク",
            },
            ensure_ascii=False,
        )


class HighScoreProvider(LLMProvider):
    """Mock LLM that returns an alignment score above cap."""

    def generate(self, system: str, user: str) -> str:
        return json.dumps(
            {
                "divergences": [],
                "alignment_score": 0.98,
                "caution": "完全一致に見えるが注意",
            }
        )


class NoDivergenceProvider(LLMProvider):
    """Mock LLM that returns no divergences and no caution."""

    def generate(self, system: str, user: str) -> str:
        return json.dumps(
            {
                "divergences": [],
                "alignment_score": 0.80,
            }
        )


def load_sample_responses():
    with open(FIXTURES / "sample_responses.json") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


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
    assert result["alignment_score"] == 0.55


def test_alignment_score_cap():
    """PRD 4.1: alignment_score must be capped at 0.90."""
    provider = HighScoreProvider()
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
