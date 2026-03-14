"""FastAPI application factory."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import Engine

from minddiff.db import create_tables, get_engine, get_session_factory

# Source tree root for templates/static
SRC_DIR = Path(__file__).resolve().parent


def create_app(engine: Engine | None = None) -> FastAPI:
    app = FastAPI(title="MindDiff", version="0.1.0")

    engine = engine or get_engine()
    create_tables(engine)
    app.state.session_factory = get_session_factory(engine)

    static_dir = SRC_DIR / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    templates_dir = SRC_DIR / "templates"
    if templates_dir.is_dir():
        app.state.templates = Jinja2Templates(directory=templates_dir)

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    from minddiff.routes import teams, cycles, responses, reports, pages

    app.include_router(pages.router)
    app.include_router(teams.router)
    app.include_router(cycles.router)
    app.include_router(responses.router)
    app.include_router(reports.router)

    return app
