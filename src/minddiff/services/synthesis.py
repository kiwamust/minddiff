"""Stage 1: Synthesis — integrate team responses per dimension."""

import json

from minddiff.services.llm import LLMProvider

DIMENSION_LABELS = {
    1: "目的理解",
    2: "状況認識",
    3: "リスク認知",
    4: "合意事項",
    5: "優先順位",
}

SYSTEM_PROMPT = """\
あなたはチームの認識を統合するアナリストです。

## 原則
- メンバーの入力テキストを忠実に統合する。意味を変えない
- 各メンバーの表現を可能な限り原文に近い形で保持する
- 共通して言及されているテーマを特定する
- 言及の分布を記録する（何人中何人が言及したか）
- 入力テキストに根拠がない情報を推測・創作しない

## 出力形式（JSON）
{
  "summary": "チーム全体の認識を統合した要約（2-3文）",
  "common_themes": ["共通テーマ1", "共通テーマ2"],
  "mention_distribution": {"テーマ1": "3/5名", "テーマ2": "2/5名"},
  "notable_expressions": ["印象的なメンバーの表現を原文引用"]
}
"""


def build_user_prompt(dimension: int, responses: list[dict]) -> str:
    """Build user prompt from member responses for a single dimension."""
    label = DIMENSION_LABELS[dimension]
    lines = [f"## 次元: {label}\n"]
    for i, r in enumerate(responses, 1):
        lines.append(f"### メンバー{i}\n{r['content']}\n")
    lines.append("上記の回答を統合してください。JSON形式で出力してください。")
    return "\n".join(lines)


def synthesize_dimension(
    provider: LLMProvider,
    dimension: int,
    responses: list[dict],
) -> dict:
    """Run Stage 1 synthesis for a single dimension."""
    user_prompt = build_user_prompt(dimension, responses)
    raw = provider.generate(SYSTEM_PROMPT, user_prompt)

    # Extract JSON from response (handle markdown code blocks)
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]

    return json.loads(text)


def synthesize_all(
    provider: LLMProvider,
    responses_by_dimension: dict[int, list[dict]],
) -> dict:
    """Run synthesis for all 5 dimensions."""
    result = {}
    for dim in range(1, 6):
        resps = responses_by_dimension.get(dim, [])
        if resps:
            result[str(dim)] = synthesize_dimension(provider, dim, resps)
        else:
            result[str(dim)] = {
                "summary": "回答なし",
                "common_themes": [],
                "mention_distribution": {},
            }
    return result
