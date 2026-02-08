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
        description="SVG spatial analysis engine — geometry transforms for LLM comprehension",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import all transform modules to trigger registration
    _register_transforms()

    from app.api.router import api_router

    app.include_router(api_router)

    return app


def _register_transforms() -> None:
    """Import all transform modules so @transform decorators fire."""
    import importlib
    import pkgutil

    for layer_name in ["layer0", "layer1", "layer2", "layer3", "layer4"]:
        package_name = f"app.engine.{layer_name}"
        try:
            package = importlib.import_module(package_name)
            for _, module_name, _ in pkgutil.iter_modules(package.__path__):
                importlib.import_module(f"{package_name}.{module_name}")
        except ModuleNotFoundError:
            pass


app = create_app()
