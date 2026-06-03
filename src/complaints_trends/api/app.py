from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import auth, gigachat, health, records


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
    app.include_router(auth.router)
    app.include_router(gigachat.router)
    app.include_router(records.router)

    return app
