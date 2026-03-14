"""Test synthesis service."""

import json
from pathlib import Path

from minddiff.services.synthesis import build_user_prompt, synthesize_dimension, synthesize_all
from minddiff.services.llm import LLMProvider


FIXTURES = Path(__file__).parent / "fixtures"


class MockProvider(LLMProvider):
    """Mock LLM that returns predetermined synthesis results."""

    def generate(self, system: str, user: str) -> str:
        return json.dumps(
            {
                "summary": "チーム全体でQ1リリースに向けた機能X実装が最優先。ただし品質水準に認識差あり。",
                "common_themes": ["Q1リリース", "機能X", "品質基準"],
                "mention_distribution": {
                    "Q1リリース": "3/3名",
                    "品質基準": "2/3名",
                },
                "notable_expressions": [
                    "品質は後から上げればいい",
                    "本番品質まで仕上げる前提",
                ],
            },
            ensure_ascii=False,
        )


def load_sample_responses():
    with open(FIXTURES / "sample_responses.json") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def test_build_user_prompt():
    responses = [
        {"content": "Q1リリースが最優先"},
        {"content": "機能Xを完成させる"},
    ]
    prompt = build_user_prompt(1, responses)
    assert "目的理解" in prompt
    assert "メンバー1" in prompt
    assert "メンバー2" in prompt
    assert "Q1リリースが最優先" in prompt


def test_synthesize_dimension():
    provider = MockProvider()
    responses = load_sample_responses()
    result = synthesize_dimension(provider, 1, responses[1])

    assert "summary" in result
    assert "common_themes" in result
    assert isinstance(result["common_themes"], list)
    assert "mention_distribution" in result


def test_synthesize_all():
    provider = MockProvider()
    responses = load_sample_responses()
    result = synthesize_all(provider, responses)

    assert len(result) == 5
    for dim in range(1, 6):
        assert str(dim) in result
        assert "summary" in result[str(dim)]


def test_synthesize_handles_code_block():
    """Test that synthesis can handle markdown code blocks in LLM output."""

    class CodeBlockProvider(LLMProvider):
        def generate(self, system: str, user: str) -> str:
            return (
                '```json\n{"summary": "test", "common_themes": [], "mention_distribution": {}}\n```'
            )

    provider = CodeBlockProvider()
    result = synthesize_dimension(provider, 1, [{"content": "test"}])
    assert result["summary"] == "test"


def test_synthesize_empty_dimension():
    provider = MockProvider()
    result = synthesize_all(provider, {})

    for dim in range(1, 6):
        assert result[str(dim)]["summary"] == "回答なし"
