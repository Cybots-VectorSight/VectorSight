"""Streaming LLM responses with extended thinking via SSE."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from app.config import settings
from app.llm.model_router import get_model_for_task
from app.llm.prompts import get_prompt_template


async def stream_chat_response(
    svg: str,
    enrichment: str,
    question: str,
    history: list[dict[str, str]],
    task: str = "chat",
) -> AsyncGenerator[str, None]:
    """Stream SSE events with separate thinking and response blocks."""
    if not settings.anthropic_api_key:
        data = json.dumps({"type": "response", "content": "[LLM not configured — set ANTHROPIC_API_KEY in .env]"})
        yield f"event: response\ndata: {data}\n\n"
        yield f"event: done\ndata: {json.dumps({'type': 'done'})}\n\n"
        return

    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    model_id = get_model_for_task(task)
    llm = ChatAnthropic(
        model=model_id,
        api_key=settings.anthropic_api_key,
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 10000},
    )

    template = get_prompt_template(task)
    system_msg = template.format(svg=svg, enrichment=enrichment)

    messages: list = [SystemMessage(content=system_msg)]
    for msg in history:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))

    try:
        async for chunk in llm.astream(messages):
            if isinstance(chunk.content, list):
                for block in chunk.content:
                    block_type = block.get("type", "")
                    if block_type == "thinking":
                        thinking_text = block.get("thinking", "")
                        if thinking_text:
                            data = json.dumps({"type": "thinking", "content": thinking_text})
                            yield f"event: thinking\ndata: {data}\n\n"
                    elif block_type == "text":
                        text = block.get("text", "")
                        if text:
                            data = json.dumps({"type": "response", "content": text})
                            yield f"event: response\ndata: {data}\n\n"
            elif isinstance(chunk.content, str) and chunk.content:
                data = json.dumps({"type": "response", "content": chunk.content})
                yield f"event: response\ndata: {data}\n\n"
    except Exception as e:
        data = json.dumps({"type": "error", "content": str(e)})
        yield f"event: error\ndata: {data}\n\n"

    yield f"event: done\ndata: {json.dumps({'type': 'done'})}\n\n"
