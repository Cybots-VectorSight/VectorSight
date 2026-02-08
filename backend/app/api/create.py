"""POST /api/create — icon creation from text description."""

from __future__ import annotations

from fastapi import APIRouter

from app.models.requests import CreateRequest
from app.models.responses import CreateResponse

router = APIRouter()


@router.post("/create", response_model=CreateResponse)
async def create(req: CreateRequest) -> CreateResponse:
    from app.llm.client import get_chat_response

    result = await get_chat_response(
        svg="",
        enrichment="",
        question=req.description,
        history=[],
        task="create",
    )

    return CreateResponse(svg=result, intent=req.description)
