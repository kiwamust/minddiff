# MindDiff

チームのメンタルモデルに `git diff` を生成する。

メンバーが持つ認識（目的・状況・リスク・合意・優先順位）を定期収集し、LLM が統合・乖離検出することで、見えない認知ギャップを可視化する。

## Core Concept

5次元フレームワークで認識を構造化し、二段階LLMパイプライン（統合 → 乖離検出）で分析する。

**最重要原則：偽の収束を防ぐ。** アラインメントスコアは最大 0.90 にキャップ。乖離の過検出は許容し、見逃しを最小化する。

## Quick Start

```bash
# 依存インストール
uv sync --extra dev

# 開発サーバー起動
cp .env.example .env
# .env の ANTHROPIC_API_KEY を設定
uv run uvicorn minddiff.app:create_app --factory --port 8000 --reload

# テストデータ投入
uv run python scripts/seed.py

# テスト実行
uv run python -m pytest tests/ -v
```

`http://localhost:8000/login` にアクセスし、seed スクリプトが出力するトークンでログイン。

## Tech Stack

| Layer    | Choice                                |
| -------- | ------------------------------------- |
| Backend  | Python 3.11+ / FastAPI / SQLAlchemy   |
| Frontend | Jinja2 + htmx + Tailwind CSS          |
| LLM      | Claude API (抽象化レイヤーで交換可能) |
| DB       | SQLite (WAL mode)                     |
| Deploy   | Fly.io (Dockerfile)                   |

## Architecture

```
Web UI (htmx) → FastAPI → SQLite
                   ↕
              Claude API
         (Synthesis + Divergence)
```

モノリス。PoC の目的はアーキテクチャの検証ではなく、仮説の検証。

## 5 Dimensions

| #   | Dimension  | Signal           |
| --- | ---------- | ---------------- |
| 1   | 目的理解   | 目的の乖離       |
| 2   | 状況認識   | 進捗認識の乖離   |
| 3   | リスク認知 | リスク認知の乖離 |
| 4   | 合意事項   | 合意事項の乖離   |
| 5   | 優先順位   | 優先順位の乖離   |

## License

[MIT](LICENSE)
