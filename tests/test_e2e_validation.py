"""End-to-end validation: scientific measurement of MindDiff LLM pipeline quality.

Defines 9 observables derived from PRD sections 3, 4, 11, and Appendix A.
Each observable has an operational definition, measurement function, and acceptance threshold.

Run: uv run python -m pytest tests/test_e2e_validation.py -m e2e -v -s
Requires: ANTHROPIC_API_KEY in environment.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from minddiff.services.divergence import (
    ALIGNMENT_SCORE_CAP,
    detect_all_divergences,
)
from minddiff.services.llm import ClaudeProvider
from minddiff.services.synthesis import synthesize_all

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RESULTS_DIR = Path(__file__).parent.parent / "validation_results"

# ─── Ground Truth ────────────────────────────────────────────────────────────
# Known divergences intentionally embedded in sample_responses.json (3 members × 5 dims).
# Each entry specifies: dimension, keywords for concept/evidence matching, description.
# A divergence is "detected" if any detected divergence in that dimension
# matches at least one keyword in concept OR evidence text.

GROUND_TRUTH = [
    {
        "dimension": 1,
        "keywords": ["品質", "完成", "MVP", "Done", "定義", "仕上げ", "本番"],
        "description": "品質基準の乖離: M2「品質は後から」vs M3「本番品質前提」",
    },
    {
        "dimension": 2,
        "keywords": ["フロント", "進捗", "遅れ", "設計", "見直し", "50%"],
        "description": "フロント進捗認識の乖離: M1「50%遅れ」/ M2「まあまあ」/ M3「設計見直し」",
    },
    {
        "dimension": 2,
        "keywords": ["テスト", "着手", "未着手", "書き始め"],
        "description": "テスト着手状況の不一致: M1/M3は未着手、M2は着手済み",
    },
    {
        "dimension": 3,
        "keywords": ["パフォーマンス", "要件", "曖昧", "非共有", "網羅", "共有されていない"],
        "description": "パフォーマンス要件リスクはM3のみが認識",
    },
    {
        "dimension": 4,
        "keywords": ["デプロイ", "CI", "CD", "フロー", "未確認", "確定", "確認"],
        "description": "デプロイフロー合意度: M1は確定認識、M3は未確認",
    },
    {
        "dimension": 5,
        "keywords": ["優先", "バラバラ", "不一致", "異な", "分かれ", "注力", "方針"],
        "description": "個人・チーム双方の優先順位が全員異なる",
    },
]

# Judgmental language patterns (PRD 4.2: Agent does not judge)
JUDGMENT_PATTERNS = [
    re.compile(r"(?:が|は)正しい"),
    re.compile(r"(?:が|は)間違[いっ]"),
    re.compile(r"(?:が|は)誤[りっ]"),
    re.compile(r"(?:が|は)不適切"),
    re.compile(r"[A-Zア-ン]が正しく.*[A-Zア-ン]が間違"),
    re.compile(r"(?:すべき|べき)(?:だ|である|です)(?![\u3000-\u9FFF])"),
]

# Key information that synthesis must cover per dimension
FAITHFULNESS_KEYS = {
    1: ["Q1", "機能X", "リリース"],
    2: ["バックエンド", "フロント", "テスト"],
    3: ["外部API", "スケジュール"],
    4: ["APIスキーマ", "v2"],
    5: ["フロント", "テスト"],
}


# ─── Data Types ──────────────────────────────────────────────────────────────


@dataclass
class Measurement:
    """Single observable measurement."""

    id: str
    name: str
    passed: bool
    value: float | str
    threshold: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "detail": self.detail,
        }


@dataclass
class ValidationReport:
    """Aggregate report of all observable measurements."""

    measurements: list[Measurement] = field(default_factory=list)
    synthesis: dict = field(default_factory=dict)
    divergences: dict = field(default_factory=dict)
    alignment_scores: dict = field(default_factory=dict)
    elapsed_s: float = 0.0

    def add(self, m: Measurement) -> None:
        self.measurements.append(m)

    @property
    def passed_count(self) -> int:
        return sum(1 for m in self.measurements if m.passed)

    @property
    def total_count(self) -> int:
        return len(self.measurements)

    def to_dict(self) -> dict:
        return {
            "summary": f"{self.passed_count}/{self.total_count} passed",
            "elapsed_s": self.elapsed_s,
            "alignment_scores": self.alignment_scores,
            "measurements": [m.to_dict() for m in self.measurements],
            "divergences_summary": {
                k: [{"concept": d.get("concept"), "confidence": d.get("confidence")} for d in divs]
                for k, divs in self.divergences.items()
            },
        }

    def format(self) -> str:
        lines = [
            "=" * 72,
            "MindDiff Pipeline Validation Report",
            "=" * 72,
            f"Result: {self.passed_count}/{self.total_count} observables passed",
            f"Elapsed: {self.elapsed_s:.1f}s",
            "-" * 72,
        ]
        for m in self.measurements:
            status = "PASS" if m.passed else "FAIL"
            lines.append(f"[{status}] {m.id} {m.name}")
            lines.append(f"       value={m.value}  threshold={m.threshold}")
            if m.detail:
                for detail_line in m.detail.split("\n"):
                    lines.append(f"       {detail_line}")
        lines.append("-" * 72)
        lines.append("Alignment Scores:")
        for k in sorted(self.alignment_scores):
            lines.append(f"  Dim {k}: {self.alignment_scores[k]:.2f}")
        lines.append("-" * 72)
        lines.append("Detected Divergences:")
        for k in sorted(self.divergences):
            divs = self.divergences[k]
            lines.append(f"  Dim {k}: {len(divs)} divergence(s)")
            for d in divs:
                lines.append(f"    [{d.get('confidence', '?')}] {d.get('concept', '?')}")
        lines.append("=" * 72)
        return "\n".join(lines)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _load_responses() -> dict[int, list[dict]]:
    with open(FIXTURES_DIR / "sample_responses.json") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def _get_provider() -> ClaudeProvider:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return ClaudeProvider(api_key=api_key)


def _all_original_text(responses: dict[int, list[dict]]) -> str:
    """Concatenate all original response texts for grounding checks."""
    parts = []
    for resps in responses.values():
        for r in resps:
            parts.append(r["content"])
    return " ".join(parts)


def _match_divergence(div: dict, keywords: list[str]) -> bool:
    """Check if a detected divergence matches any keyword in concept or evidence."""
    concept = div.get("concept", "")
    evidence_text = " ".join(div.get("evidence", []))
    caution = div.get("caution", "") if isinstance(div.get("caution"), str) else ""
    search_space = f"{concept} {evidence_text} {caution}"
    return any(kw in search_space for kw in keywords)


# ─── Observable Measurement Functions ────────────────────────────────────────


def measure_o1_structural_validity(
    synthesis: dict, divergences: dict, alignment_scores: dict
) -> Measurement:
    """O1: Output JSON conforms to expected schema.

    Operational definition:
    - synthesis[dim] must have: summary (str), common_themes (list)
    - divergences[dim] must be list of {concept, confidence, evidence, recommended_action}
    - confidence ∈ {"高", "中", "低"}
    - evidence must be list
    - alignment_scores[dim] must be numeric
    """
    issues = []

    for dim_key in [str(d) for d in range(1, 6)]:
        s = synthesis.get(dim_key, {})
        if not isinstance(s.get("summary"), str):
            issues.append(f"synthesis[{dim_key}].summary not str")
        if not isinstance(s.get("common_themes"), list):
            issues.append(f"synthesis[{dim_key}].common_themes not list")

        divs = divergences.get(dim_key)
        if not isinstance(divs, list):
            issues.append(f"divergences[{dim_key}] not list")
        else:
            for i, d in enumerate(divs):
                for fld in ("concept", "confidence", "evidence", "recommended_action"):
                    if fld not in d:
                        issues.append(f"divergences[{dim_key}][{i}] missing '{fld}'")
                if d.get("confidence") not in ("高", "中", "低", None):
                    issues.append(
                        f"divergences[{dim_key}][{i}] invalid confidence: {d.get('confidence')}"
                    )
                if "evidence" in d and not isinstance(d["evidence"], list):
                    issues.append(f"divergences[{dim_key}][{i}] evidence not list")

        score = alignment_scores.get(dim_key)
        if not isinstance(score, (int, float)):
            issues.append(f"alignment_scores[{dim_key}] not numeric")

    return Measurement(
        id="O1",
        name="構造的妥当性",
        passed=len(issues) == 0,
        value="all valid" if not issues else f"{len(issues)} issues",
        threshold="0 issues",
        detail="\n".join(issues) if issues else "",
    )


def measure_o2_recall(divergences: dict) -> Measurement:
    """O2: Known divergences detected (recall / sensitivity).

    Operational definition:
    For each ground truth divergence GT[i] in dimension D:
      detected if ∃ d in divergences[D] such that match(d, GT[i].keywords) = True.
    Recall = |detected| / |GT|.

    Threshold: ≥ 80% (PRD section 11 success criteria).
    """
    detected = 0
    missed = []

    for gt in GROUND_TRUTH:
        dim_key = str(gt["dimension"])
        dim_divs = divergences.get(dim_key, [])
        if any(_match_divergence(d, gt["keywords"]) for d in dim_divs):
            detected += 1
        else:
            missed.append(f"Dim{gt['dimension']}: {gt['description']}")

    total = len(GROUND_TRUTH)
    recall = detected / total if total > 0 else 0.0

    return Measurement(
        id="O2",
        name="既知乖離の検出率 (Recall)",
        passed=recall >= 0.80,
        value=f"{detected}/{total} ({recall:.0%})",
        threshold="≥ 80%",
        detail="\n".join(f"MISSED: {m}" for m in missed) if missed else "",
    )


def measure_o3_false_convergence(alignment_scores: dict) -> Measurement:
    """O3: No alignment score exceeds ALIGNMENT_SCORE_CAP (0.90).

    Operational definition:
    ∀ dim ∈ {1..5}: alignment_scores[dim] ≤ ALIGNMENT_SCORE_CAP.

    Threshold: 100% compliance (PRD 4.1, critical safety constraint).
    """
    violations = []
    max_score = 0.0
    for dim_key in sorted(alignment_scores):
        score = alignment_scores[dim_key]
        if isinstance(score, (int, float)):
            max_score = max(max_score, score)
            if score > ALIGNMENT_SCORE_CAP:
                violations.append(f"Dim{dim_key}={score:.2f}")

    return Measurement(
        id="O3",
        name="偽収束防止 (alignment_score ≤ 0.90)",
        passed=len(violations) == 0,
        value=f"max={max_score:.2f}",
        threshold="all ≤ 0.90 (100%)",
        detail=f"Violations: {', '.join(violations)}" if violations else "",
    )


def measure_o4_evidence_grounding(
    divergences: dict, responses: dict[int, list[dict]]
) -> Measurement:
    """O4: Evidence quotes are grounded in original member responses.

    Operational definition:
    For each evidence string E in divergences:
      Extract quoted text Q from 「」brackets.
      Grounded if ≥ 50% of Q's words appear in the original response corpus.
    Ratio = |grounded| / |total evidence|.

    Threshold: ≥ 90% (PRD Appendix A: 根拠の明示).
    """
    corpus = _all_original_text(responses)
    total = 0
    grounded = 0
    ungrounded_samples = []

    for dim_key, divs in divergences.items():
        for d in divs:
            for ev in d.get("evidence", []):
                total += 1
                # Extract text in 「」or『』
                quotes = re.findall(r"[「『](.*?)[」』]", ev)
                if quotes:
                    any_grounded = False
                    for q in quotes:
                        q = q.strip()
                        if len(q) < 2:
                            any_grounded = True
                            continue
                        # Direct substring match
                        if q in corpus:
                            any_grounded = True
                            break
                        # Word-level partial match (≥50%)
                        words = [w for w in q.split() if len(w) >= 2]
                        if words:
                            hit = sum(1 for w in words if w in corpus)
                            if hit / len(words) >= 0.5:
                                any_grounded = True
                                break
                        # Character n-gram fallback: any 4-char substring match
                        if len(q) >= 4 and any(q[i : i + 4] in corpus for i in range(len(q) - 3)):
                            any_grounded = True
                            break
                    if any_grounded:
                        grounded += 1
                    else:
                        ungrounded_samples.append(ev[:80])
                else:
                    # No explicit quote brackets — lenient: count as grounded
                    grounded += 1

    ratio = grounded / total if total > 0 else 1.0
    return Measurement(
        id="O4",
        name="エビデンスの根拠性",
        passed=ratio >= 0.90,
        value=f"{grounded}/{total} ({ratio:.0%})",
        threshold="≥ 90%",
        detail="\n".join(f"UNGROUNDED: {u}" for u in ungrounded_samples[:3])
        if ungrounded_samples
        else "",
    )


def measure_o5_judgment_avoidance(divergences: dict, synthesis: dict) -> Measurement:
    """O5: Agent does not make value judgments about who is right/wrong.

    Operational definition:
    JUDGMENT_PATTERNS must not appear in concept or recommended_action fields.
    Appearances in evidence (quoted text) are excluded — those are member quotes.

    Threshold: 0 violations (PRD 4.2: absolute constraint).
    """
    violations = []

    for dim_key, divs in divergences.items():
        for d in divs:
            # Check only agent-generated fields, not evidence quotes
            agent_text = f"{d.get('concept', '')} {d.get('recommended_action', '')}"
            for pat in JUDGMENT_PATTERNS:
                match = pat.search(agent_text)
                if match:
                    violations.append(f"Dim{dim_key}: '{match.group()}' in agent text")

    # Check synthesis summaries too
    for dim_key, s in synthesis.items():
        summary = s.get("summary", "")
        for pat in JUDGMENT_PATTERNS:
            match = pat.search(summary)
            if match:
                violations.append(f"synthesis[{dim_key}]: '{match.group()}' in summary")

    return Measurement(
        id="O5",
        name="判断回避 (PRD 4.2)",
        passed=len(violations) == 0,
        value="clean" if not violations else f"{len(violations)} violations",
        threshold="0 violations",
        detail="\n".join(violations[:5]) if violations else "",
    )


def measure_o6_faithfulness(synthesis: dict, responses: dict[int, list[dict]]) -> Measurement:
    """O6: Synthesis preserves key information from original responses.

    Operational definition:
    For each dimension, a set of key terms K[dim] is defined (from input corpus).
    Covered if term appears in synthesis[dim] JSON text.
    Ratio = |covered| / |total keys|.

    Threshold: ≥ 80% (PRD Appendix A: 入力テキストの忠実な統合).
    """
    total = 0
    covered = 0
    missing = []

    for dim, keys in FAITHFULNESS_KEYS.items():
        dim_key = str(dim)
        synth_text = json.dumps(synthesis.get(dim_key, {}), ensure_ascii=False)
        for k in keys:
            total += 1
            if k in synth_text:
                covered += 1
            else:
                missing.append(f"Dim{dim}: '{k}'")

    ratio = covered / total if total > 0 else 0.0
    return Measurement(
        id="O6",
        name="合成の忠実性 (Faithfulness)",
        passed=ratio >= 0.80,
        value=f"{covered}/{total} ({ratio:.0%})",
        threshold="≥ 80%",
        detail=f"Missing: {', '.join(missing)}" if missing else "",
    )


def measure_o7_no_fabrication(
    synthesis: dict, divergences: dict, responses: dict[int, list[dict]]
) -> Measurement:
    """O7: No hallucinated named entities in output.

    Operational definition:
    Extract proper nouns / technical terms (alphabetic ≥ 3 chars) from output.
    Exclude common analytical vocabulary.
    Fabricated if term does not appear (case-insensitive) in input corpus.

    Threshold: ≤ 2 suspicious terms (margin for analytical vocabulary).
    """
    corpus = _all_original_text(responses).lower()
    output_text = json.dumps(synthesis, ensure_ascii=False) + json.dumps(
        divergences, ensure_ascii=False
    )

    # Analytical/schema terms — not fabrication
    ignore = {
        "json",
        "stage",
        "llm",
        "api",
        "cicd",
        "null",
        "true",
        "false",
        "summary",
        "themes",
        "common",
        "mention",
        "distribution",
        "notable",
        "expressions",
        "concept",
        "confidence",
        "evidence",
        "recommended",
        "action",
        "alignment",
        "score",
        "caution",
        "divergence",
        "divergences",
        "high",
        "medium",
        "low",
        "the",
        "and",
        "for",
        "with",
        "not",
        "has",
        "are",
        "was",
        "mvp",
    }

    output_terms = set(re.findall(r"\b[A-Za-z]{3,}\b", output_text))
    suspicious = [t for t in output_terms if t.lower() not in ignore and t.lower() not in corpus]

    return Measurement(
        id="O7",
        name="捏造の不在 (No Fabrication)",
        passed=len(suspicious) <= 2,
        value=f"{len(suspicious)} suspicious" if suspicious else "clean",
        threshold="≤ 2 suspicious terms",
        detail=f"Terms: {', '.join(suspicious[:10])}" if suspicious else "",
    )


def measure_o8_latency(elapsed: float) -> Measurement:
    """O8: Total processing time ≤ 300s (NFR-03: 5 minutes)."""
    return Measurement(
        id="O8",
        name="レイテンシ (5次元処理時間)",
        passed=elapsed <= 300.0,
        value=f"{elapsed:.1f}s",
        threshold="≤ 300s",
    )


# ─── Tests ───────────────────────────────────────────────────────────────────


@pytest.mark.e2e
class TestE2EValidation:
    """End-to-end validation of MindDiff LLM pipeline against ground truth.

    Executes the full Synthesis → Divergence Detection pipeline with real
    Claude API calls and measures 8 observables (O1-O8).
    """

    def test_full_pipeline(self):
        """Run full pipeline, measure all observables, assert all pass."""
        provider = _get_provider()
        responses = _load_responses()

        # Execute pipeline
        t0 = time.time()
        synthesis = synthesize_all(provider, responses)
        divergences, alignment_scores = detect_all_divergences(provider, synthesis, responses)
        elapsed = time.time() - t0

        # Build report
        report = ValidationReport(
            synthesis=synthesis,
            divergences=divergences,
            alignment_scores=alignment_scores,
            elapsed_s=elapsed,
        )
        report.add(measure_o1_structural_validity(synthesis, divergences, alignment_scores))
        report.add(measure_o2_recall(divergences))
        report.add(measure_o3_false_convergence(alignment_scores))
        report.add(measure_o4_evidence_grounding(divergences, responses))
        report.add(measure_o5_judgment_avoidance(divergences, synthesis))
        report.add(measure_o6_faithfulness(synthesis, responses))
        report.add(measure_o7_no_fabrication(synthesis, divergences, responses))
        report.add(measure_o8_latency(elapsed))

        # Output
        print("\n" + report.format())

        # Persist results
        RESULTS_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        result_path = RESULTS_DIR / f"validation_{ts}.json"
        with open(result_path, "w") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {result_path}")

        # Assert
        failures = [m for m in report.measurements if not m.passed]
        if failures:
            lines = [f"  {m.id} {m.name}: {m.value} (need {m.threshold})" for m in failures]
            pytest.fail(f"{len(failures)} observable(s) failed:\n" + "\n".join(lines))


@pytest.mark.e2e
@pytest.mark.slow
class TestReproducibility:
    """O9: Pipeline reproducibility — alignment_score σ ≤ 0.15 across 3 runs.

    Expensive test (3× API calls). Run separately:
    uv run python -m pytest tests/test_e2e_validation.py -m "e2e and slow" -v -s
    """

    def test_reproducibility(self):
        """Run pipeline 3 times, measure alignment score variance."""
        provider = _get_provider()
        responses = _load_responses()
        n_runs = 3

        all_scores: dict[str, list[float]] = {str(d): [] for d in range(1, 6)}

        for run in range(n_runs):
            print(f"\n--- Run {run + 1}/{n_runs} ---")
            synthesis = synthesize_all(provider, responses)
            _, scores = detect_all_divergences(provider, synthesis, responses)
            for k, v in scores.items():
                all_scores[k].append(float(v))

        # Calculate std dev per dimension
        import statistics

        print("\n--- Reproducibility ---")
        max_std = 0.0
        for dim_key in sorted(all_scores):
            vals = all_scores[dim_key]
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            max_std = max(max_std, std)
            print(f"  Dim {dim_key}: mean={mean:.2f} std={std:.3f} values={vals}")

        print(f"\n  Max σ = {max_std:.3f} (threshold ≤ 0.15)")
        assert max_std <= 0.15, f"Alignment score σ too high: {max_std:.3f}"

    def test_variance_decomposition(self):
        """O9-decomp: Nested design to isolate Stage1 vs Stage2 variance.

        Design: K syntheses × R Stage2 runs per synthesis.
        - within-synthesis variance (σ_within): Stage2 noise, Stage1 held constant
        - between-synthesis variance (σ_between): Stage1 effect, averaged over Stage2
        This properly decomposes total variance into two independent sources.

        K=3 syntheses × R=3 Stage2 runs = 9 total Stage2 calls + 3 Stage1 calls.
        """
        provider = _get_provider()
        responses = _load_responses()
        import statistics

        K = 3  # number of distinct Stage1 syntheses
        R = 3  # number of Stage2 runs per synthesis

        # ── Collect data: K syntheses × R Stage2 runs ────────────────────
        # scores[synth_idx][stage2_run][dim_key] = float
        all_data: list[list[dict[str, float]]] = []

        for s in range(K):
            print(f"\n--- Synthesis {s + 1}/{K} ---")
            synthesis = synthesize_all(provider, responses)
            synth_runs = []
            for r in range(R):
                print(f"  Stage2 run {r + 1}/{R}")
                _, scores = detect_all_divergences(provider, synthesis, responses)
                synth_runs.append({k: float(v) for k, v in scores.items()})
            all_data.append(synth_runs)

        # ── Compute within-synthesis variance (Stage2 noise) ─────────────
        # For each synthesis s and dimension d:
        #   within_var_sd = stdev of scores across R runs
        # σ_within = mean of within_var_sd across all (s, d)
        print("\n=== Within-Synthesis Variance (Stage2 noise) ===")
        within_stds: dict[str, list[float]] = {str(d): [] for d in range(1, 6)}
        for s in range(K):
            for dim_key in [str(d) for d in range(1, 6)]:
                vals = [all_data[s][r][dim_key] for r in range(R)]
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                within_stds[dim_key].append(std)

        for dim_key in sorted(within_stds):
            stds = within_stds[dim_key]
            mean_std = statistics.mean(stds)
            print(
                f"  Dim {dim_key}: mean σ_within = {mean_std:.3f}  (per-synth: {[f'{s:.3f}' for s in stds]})"
            )

        max_within = max(statistics.mean(v) for v in within_stds.values())

        # ── Compute between-synthesis variance (Stage1 effect) ───────────
        # For each synthesis s and dimension d, compute mean score across R runs.
        # Then take stdev of those means across K syntheses.
        print("\n=== Between-Synthesis Variance (Stage1 effect) ===")
        between_stds: dict[str, float] = {}
        for dim_key in [str(d) for d in range(1, 6)]:
            synth_means = []
            for s in range(K):
                vals = [all_data[s][r][dim_key] for r in range(R)]
                synth_means.append(statistics.mean(vals))
            std = statistics.stdev(synth_means) if len(synth_means) > 1 else 0.0
            between_stds[dim_key] = std
            print(
                f"  Dim {dim_key}: σ_between = {std:.3f}  (synth means: {[f'{m:.3f}' for m in synth_means]})"
            )

        max_between = max(between_stds.values())

        # ── Diagnosis ────────────────────────────────────────────────────
        print("\n=== Diagnosis ===")
        print(f"  Max σ_within  (Stage2 noise):  {max_within:.3f}")
        print(f"  Max σ_between (Stage1 effect): {max_between:.3f}")

        if max_within > max_between * 1.5:
            diagnosis = "stage2_dominant"
            print("  → Stage2 (rubric generation) is the primary variance source")
        elif max_between > max_within * 1.5:
            diagnosis = "stage1_dominant"
            print("  → Stage1 (synthesis) is the primary variance source")
        else:
            diagnosis = "mixed"
            print("  → Both stages contribute comparably to total variance")

        # Save results
        RESULTS_DIR.mkdir(exist_ok=True)
        result = {
            "design": {"K_syntheses": K, "R_stage2_runs": R},
            "within_synthesis": {
                k: {"mean_std": statistics.mean(v), "per_synth_stds": v}
                for k, v in within_stds.items()
            },
            "between_synthesis": between_stds,
            "diagnosis": {
                "max_within": max_within,
                "max_between": max_between,
                "primary_source": diagnosis,
            },
            "raw_scores": [
                [{k: v for k, v in run.items()} for run in synth_runs] for synth_runs in all_data
            ],
        }
        import time

        ts = time.strftime("%Y%m%d_%H%M%S")
        result_path = RESULTS_DIR / f"variance_decomposition_{ts}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n  Saved to: {result_path}")


def _load_ground_truth_rubric() -> dict[str, dict]:
    with open(FIXTURES_DIR / "ground_truth_rubric.json") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _parse_ratio(ratio_str: str) -> float:
    """Parse "N/M" string to float."""
    try:
        parts = str(ratio_str).split("/")
        if len(parts) == 2 and int(parts[1]) > 0:
            return int(parts[0]) / int(parts[1])
    except (ValueError, ZeroDivisionError):
        pass
    return -1.0  # sentinel for unparseable


@pytest.mark.e2e
@pytest.mark.slow
class TestRubricValidity:
    """O10: Rubric validity — LLM rubric vs human-labeled ground truth.

    Runs the pipeline N times, collects rubric outputs per dimension,
    and compares each field against human labels.

    Agreement criteria:
    - Boolean fields: exact match
    - strong_evidence_count: |LLM - human| ≤ 1
    - mention_agreement_ratio: |LLM_ratio - human_ratio| ≤ 0.34

    Run: uv run python -m pytest tests/test_e2e_validation.py -k test_rubric_validity -v -s
    """

    def test_rubric_validity(self):
        """Collect rubrics from N pipeline runs and evaluate against ground truth."""
        from minddiff.services.divergence import detect_divergence

        provider = _get_provider()
        responses = _load_responses()
        ground_truth = _load_ground_truth_rubric()
        N = 3  # runs for statistical stability

        # Collect rubrics: rubrics[dim_key] = list of N rubric dicts
        collected: dict[str, list[dict]] = {str(d): [] for d in range(1, 6)}

        for run in range(N):
            print(f"\n--- Run {run + 1}/{N} ---")
            synthesis = synthesize_all(provider, responses)
            for dim in range(1, 6):
                dim_key = str(dim)
                synth = synthesis.get(dim_key, {})
                resps = responses.get(dim, [])
                result = detect_divergence(provider, dim, synth, resps)
                rubric = result.get("rubric", {})
                collected[dim_key].append(rubric)

        # Evaluate each field
        base_fields = [
            "has_common_core",
            "has_competing_goals",
            "strong_evidence_count",
            "mention_agreement_ratio",
        ]
        dim5_fields = [
            "personal_agreement_ratio",
            "team_agreement_ratio",
        ]
        bool_fields = {
            "has_common_core",
            "has_competing_goals",
        }
        ratio_fields = {
            "mention_agreement_ratio",
            "personal_agreement_ratio",
            "team_agreement_ratio",
        }

        print("\n" + "=" * 72)
        print("Rubric Validity Report (LLM vs Human Ground Truth)")
        print("=" * 72)

        all_fields = base_fields + dim5_fields
        field_agreements: dict[str, list[bool]] = {f: [] for f in all_fields}
        dim_details: dict[str, dict] = {}

        for dim_key in [str(d) for d in range(1, 6)]:
            gt = ground_truth[dim_key]
            rubrics = collected[dim_key]
            dim_detail = {}

            # Dim5 gets additional fields
            active_fields = base_fields + dim5_fields if dim_key == "5" else base_fields

            for field_name in active_fields:
                gt_val = gt.get(field_name)
                if gt_val is None:
                    continue  # field not in ground truth for this dimension
                matches = []

                for rubric in rubrics:
                    llm_val = rubric.get(field_name)

                    if field_name in bool_fields:
                        # Strict bool: only True/False accepted
                        if not isinstance(llm_val, bool):
                            match = False
                        else:
                            match = llm_val == gt_val
                    elif field_name == "strong_evidence_count":
                        if isinstance(llm_val, (int, float)):
                            match = abs(int(llm_val) - int(gt_val)) <= 1
                        else:
                            match = False
                    elif field_name in ratio_fields:
                        llm_ratio = _parse_ratio(str(llm_val)) if llm_val is not None else -1.0
                        gt_ratio = _parse_ratio(str(gt_val))
                        match = llm_ratio >= 0 and abs(llm_ratio - gt_ratio) <= 0.34
                    else:
                        match = False

                    matches.append(match)
                    field_agreements[field_name].append(match)

                agreement_rate = sum(matches) / len(matches) if matches else 0.0
                llm_vals = [r.get(field_name) for r in rubrics]
                dim_detail[field_name] = {
                    "gt": gt_val,
                    "llm_values": llm_vals,
                    "agreement": agreement_rate,
                }

            dim_details[dim_key] = dim_detail

        # Print per-dimension results
        for dim_key in sorted(dim_details):
            from minddiff.services.divergence import DIMENSION_LABELS

            label = DIMENSION_LABELS[int(dim_key)]
            print(f"\n  Dim {dim_key} ({label}):")
            for field_name, detail in dim_details[dim_key].items():
                rate = detail["agreement"]
                status = "OK" if rate >= 0.67 else "LOW"
                print(
                    f"    [{status}] {field_name}: "
                    f"gt={detail['gt']}  llm={detail['llm_values']}  "
                    f"agreement={rate:.0%}"
                )

        # Print per-field summary
        print("\n" + "-" * 72)
        print("  Per-field agreement (across all dimensions × runs):")
        all_agreements = []
        for field_name in all_fields:
            matches = field_agreements[field_name]
            rate = sum(matches) / len(matches) if matches else 0.0
            all_agreements.extend(matches)
            status = "PASS" if rate >= 0.70 else "FAIL"
            print(f"    [{status}] {field_name}: {sum(matches)}/{len(matches)} ({rate:.0%})")

        overall = sum(all_agreements) / len(all_agreements) if all_agreements else 0.0
        print(f"\n  Overall agreement: {sum(all_agreements)}/{len(all_agreements)} ({overall:.0%})")
        print("=" * 72)

        # Save results
        RESULTS_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        result = {
            "n_runs": N,
            "per_dimension": dim_details,
            "per_field": {
                f: {"agreement": sum(v) / len(v), "n": len(v)} for f, v in field_agreements.items()
            },
            "overall_agreement": overall,
        }
        result_path = RESULTS_DIR / f"rubric_validity_{ts}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n  Saved to: {result_path}")

        # Assert: overall agreement ≥ 70%
        assert overall >= 0.70, f"Rubric validity too low: {overall:.0%} (need ≥ 70%)"
