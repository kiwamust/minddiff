"""Stage 2: Divergence detection — find cognitive gaps between members."""

import json

from minddiff.services.llm import LLMProvider

DIMENSION_LABELS = {
    1: "目的理解",
    2: "状況認識",
    3: "リスク認知",
    4: "合意事項",
    5: "優先順位",
}

# PRD 4.1: 偽の収束を防ぐ。一致判定には慎重に、乖離検出には積極的に。
ALIGNMENT_SCORE_CAP = 0.90

SYSTEM_PROMPT = """\
あなたはチームの認識ギャップを検出するアナリストです。

## 最重要原則
- 一致の判定には慎重に。乖離の検出には積極的に
- 収束を断定するより、乖離を過検出する方が安全
- 各次元で最低1つの注意点を報告すること
- alignment_score の上限は 0.90。完全一致と判定してはならない
- 入力にない情報を推測・創作しない

## 入力
- Stage 1 の統合結果
- メンバーの原文回答

## 出力形式（JSON）
{
  "divergences": [
    {
      "concept": "乖離している概念を一言で",
      "confidence": "高" | "中" | "低",
      "evidence": [
        "メンバー①: 「原文引用」",
        "メンバー③: 「原文引用」"
      ],
      "recommended_action": "推奨アクション"
    }
  ],
  "alignment_score": 0.0〜0.90の数値,
  "caution": "この次元で注意すべき点"
}
"""


def build_user_prompt(
    dimension: int,
    synthesis: dict,
    responses: list[dict],
) -> str:
    """Build user prompt for divergence detection."""
    label = DIMENSION_LABELS[dimension]
    lines = [f"## 次元: {label}\n"]
    lines.append(f"### Stage 1 統合結果\n{json.dumps(synthesis, ensure_ascii=False, indent=2)}\n")
    lines.append("### メンバー原文回答")
    for i, r in enumerate(responses, 1):
        lines.append(f"#### メンバー{i}\n{r['content']}\n")
    lines.append(
        "上記を分析し、メンバー間の認識の乖離を検出してください。JSON形式で出力してください。"
    )
    return "\n".join(lines)


def detect_divergence(
    provider: LLMProvider,
    dimension: int,
    synthesis: dict,
    responses: list[dict],
) -> dict:
    """Run Stage 2 divergence detection for a single dimension."""
    user_prompt = build_user_prompt(dimension, synthesis, responses)
    raw = provider.generate(SYSTEM_PROMPT, user_prompt)

    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]

    result = json.loads(text)

    # Enforce alignment score cap (PRD 4.1)
    if "alignment_score" in result:
        result["alignment_score"] = min(float(result["alignment_score"]), ALIGNMENT_SCORE_CAP)

    # Ensure at least one caution point
    if not result.get("divergences") and not result.get("caution"):
        result["caution"] = "表面的には一致しているが、詳細の解釈にズレがある可能性を排除できない"

    return result


def detect_all_divergences(
    provider: LLMProvider,
    synthesis_by_dimension: dict[str, dict],
    responses_by_dimension: dict[int, list[dict]],
) -> tuple[dict, dict]:
    """Run divergence detection for all dimensions.

    Returns (divergences_dict, alignment_scores_dict).
    """
    divergences = {}
    alignment_scores = {}
    for dim in range(1, 6):
        dim_key = str(dim)
        synth = synthesis_by_dimension.get(dim_key, {})
        resps = responses_by_dimension.get(dim, [])
        if resps:
            result = detect_divergence(provider, dim, synth, resps)
            divergences[dim_key] = result.get("divergences", [])
            alignment_scores[dim_key] = result.get("alignment_score", 0.5)
        else:
            divergences[dim_key] = []
            alignment_scores[dim_key] = 0.0
    return divergences, alignment_scores
