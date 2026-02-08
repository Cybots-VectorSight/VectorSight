"""Task → model selection. Cheap models for Q&A, mid-tier for edits, frontier for creation."""

from __future__ import annotations

from app.config import settings

_TASK_MODEL_MAP = {
    "chat": "cheap",
    "analyze": "cheap",
    "modify": "mid",
    "create": "mid",
    "icon_set": "mid",
    "playground": "cheap",
}


def get_model_for_task(task: str) -> str:
    tier = _TASK_MODEL_MAP.get(task, "cheap")
    if tier == "cheap":
        return settings.model_cheap
    elif tier == "mid":
        return settings.model_mid
    else:
        return settings.model_frontier
