"""LangChain ChatAnthropic wrapper."""

from __future__ import annotations

from app.config import settings
from app.llm.model_router import get_model_for_task
from app.llm.prompts import get_prompt_template
from app.svg.anonymizer import sanitize_for_llm


async def get_chat_response(
    svg: str,
    enrichment: str,
    question: str,
    history: list[dict[str, str]],
    task: str = "chat",
) -> str:
    """Get LLM response using LangChain."""
    if not settings.anthropic_api_key:
        return "[LLM not configured — set ANTHROPIC_API_KEY in .env]"

    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    model_id = get_model_for_task(task)
    llm = ChatAnthropic(
        model=model_id,
        api_key=settings.anthropic_api_key,
        max_tokens=4096,
    )

    template = get_prompt_template(task)
    system_msg = template.format(svg=sanitize_for_llm(svg), enrichment=enrichment)

    messages: list = [SystemMessage(content=system_msg)]
    for msg in history:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=question))

    response = await llm.ainvoke(messages)
    return str(response.content)
