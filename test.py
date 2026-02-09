"""Test round 11 — enrichment has shape descriptors + behind-body annotations."""
import asyncio
import sys
import os
import time

sys.path.insert(0, r"C:\Users\Chumm\Desktop\Nomad\backend")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["VECTORSIGHT_LOG_LEVEL"] = "WARNING"

from dotenv import load_dotenv
load_dotenv(r"C:\Users\Chumm\Desktop\Nomad\backend\.env")

from app.main import _register_transforms
from app.svg.parser import parse_svg
from app.engine.pipeline import create_pipeline
from app.llm.enrichment_formatter import context_to_enrichment_text
from app.llm.prompts import get_prompt_template
from app.config import settings

_register_transforms()

with open(r"C:\Users\Chumm\Desktop\Nomad\samples\test\unknown_anon.svg", encoding="utf-8") as f:
    svg = f.read()

t0 = time.time()
print("[%.0fs] Parsing SVG (%d chars)..." % (0, len(svg)))
parsed = parse_svg(svg)
print("[%.0fs] Parsed: %d elements. Running pipeline..." % (time.time() - t0, len(parsed.subpaths)))
pipeline = create_pipeline()
ctx = pipeline.run(parsed)
print("[%.0fs] Pipeline done: %d transforms. Generating enrichment..." % (time.time() - t0, len(ctx.completed_transforms)))
enrichment = context_to_enrichment_text(ctx)
print("[%.0fs] Enrichment: %d words" % (time.time() - t0, len(enrichment.split())))

template = get_prompt_template("chat")
system_msg = template.format(svg=svg, enrichment=enrichment)

from app.learning.tags import get_spatial_reference
spatial_ref = get_spatial_reference()
if spatial_ref:
    system_msg += "\n\n=== SPATIAL PERCEPTION REFERENCE ===\n" + spatial_ref

print("[%.0fs] System message: %d chars (~%dk tokens)" % (time.time() - t0, len(system_msg), len(system_msg) // 4000))

question = """Identify what this SVG depicts. Work through these steps:

1. STRUCTURE: Check the reading hints for "Major structural elements." These are ordered by draw layer (z-index).
   - The element marked "PRIMARY OUTLINE" defines the overall shape boundary.
   - Elements marked "behind primary outline" are drawn BEHIND the main shape. They are NOT the body — they are separate appendages or features extending behind the subject (tail, wings, cape, hair, shadow, etc.). Their shape descriptor (rounded mass, compact blob, elongated, etc.) and size tell you what kind of behind-body feature they are.
   - A large "rounded mass" behind the body that covers most of the width but only the lower half suggests a large, bushy, curved appendage.

2. POSE: Read the ASCII grid. Is the figure upright, sideways, or other? Where is the widest point? Where does it taper?

3. FACING: Concentric groups marked "root-level (independent, drawn on top)" are surface features drawn over the main subject. The highest-circularity one NOT at canvas edge is likely an eye. Which side is it on?

4. FEATURES: For each concentric group:
   - "inside [element]" = belongs to that specific behind-body element (e.g., an object held in/on that appendage)
   - "root-level" = drawn on top, a facial feature or surface marking
   - "at canvas edge" = likely incidental

5. SILHOUETTE: Trace the grid outline and compare to common subjects.

6. IDENTITY: Give your TOP 3 guesses with detailed reasoning."""

# Output dir = project root
OUT_DIR = r"C:\Users\Chumm\Desktop\Nomad"


async def test_model(model_name, model_id):
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatAnthropic(
        model=model_id,
        api_key=settings.anthropic_api_key,
        max_tokens=4096,
    )

    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=question),
    ]

    print("\n[%.0fs] Calling %s (%s)..." % (time.time() - t0, model_name, model_id))
    call_start = time.time()
    response = await llm.ainvoke(messages)
    content = str(response.content)
    call_time = time.time() - call_start

    out = os.path.join(OUT_DIR, "response_%s.txt" % model_name)
    with open(out, "w", encoding="utf-8") as f:
        f.write(content)
    print("[%.0fs] %s done in %.0fs: %d chars -> %s" % (time.time() - t0, model_name, call_time, len(content), out))
    return content


async def main():
    sonnet = await test_model("sonnet", settings.model_mid)
    haiku = await test_model("haiku", settings.model_cheap)
    print("\n[%.0fs] ALL DONE" % (time.time() - t0))


asyncio.run(main())
