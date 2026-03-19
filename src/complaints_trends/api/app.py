from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .deps import get_services
from .routers import categories, feedback, health, meta, overview, pattern_fit, pattern_monitor, preparation, reports, runs, timeseries


def create_app(config_path: str) -> FastAPI:
    app = FastAPI(title="complaints-trends dashboard api", version="0.1.0")
    app.state.config_path = config_path

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(meta.router)
    app.include_router(overview.router)
    app.include_router(categories.router)
    app.include_router(timeseries.router)
    app.include_router(pattern_fit.router)
    app.include_router(pattern_monitor.router)
    app.include_router(feedback.router)
    app.include_router(preparation.router)
    app.include_router(reports.router)
    app.include_router(runs.router)

    return app

