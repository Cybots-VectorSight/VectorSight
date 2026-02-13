"""FastAPI app factory."""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

load_dotenv()

logging.basicConfig(
    level=getattr(logging, settings.vectorsight_log_level.upper(), logging.DEBUG),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="VectorSight",
        description="SVG spatial analysis engine -- 3-stage breakdown pipeline for LLM comprehension",
        version="0.2.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from app.api.router import api_router

    app.include_router(api_router)

    return app


app = create_app()
