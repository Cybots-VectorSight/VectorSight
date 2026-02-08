"""Master API router — mounts all endpoint routers."""

from __future__ import annotations

from fastapi import APIRouter

from app.api import analyze, chat, create, health, icon_set, modify, playground

api_router = APIRouter(prefix="/api")

api_router.include_router(health.router)
api_router.include_router(analyze.router)
api_router.include_router(chat.router)
api_router.include_router(modify.router)
api_router.include_router(create.router)
api_router.include_router(icon_set.router)
api_router.include_router(playground.router)
