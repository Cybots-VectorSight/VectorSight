"""Test pixel segmentation pipeline on sample SVGs."""

import re
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.engine.pixel_segmentation import (
    run_pixel_pipeline,
    format_result_text,
    format_enrichment_sections,
)

SAMPLES = {
    "Smiley": """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="50" cy="50" r="45" fill="#FFD700"/>
  <circle cx="35" cy="40" r="6" fill="#333"/>
  <circle cx="65" cy="40" r="6" fill="#333"/>
  <path d="M 30 65 Q 50 85 70 65" stroke="#333" stroke-width="3" fill="none"/>
</svg>""",

    "Settings Gear": """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 15.5A3.5 3.5 0 0 1 8.5 12 3.5 3.5 0 0 1 12 8.5a3.5 3.5 0 0 1 3.5 3.5 3.5 3.5 0 0 1-3.5 3.5m7.43-2.53c.04-.32.07-.64.07-.97 0-.33-.03-.66-.07-1l2.11-1.63c.19-.15.24-.42.12-.64l-2-3.46c-.12-.22-.39-.3-.61-.22l-2.49 1c-.52-.4-1.08-.73-1.69-.98l-.38-2.65C14.46 2.18 14.25 2 14 2h-4c-.25 0-.46.18-.49.42l-.38 2.65c-.61.25-1.17.59-1.69.98l-2.49-1c-.23-.09-.49 0-.61.22l-2 3.46c-.13.22-.07.49.12.64L4.57 11c-.04.34-.07.67-.07 1 0 .33.03.65.07.97l-2.11 1.66c-.19.15-.25.42-.12.64l2 3.46c.12.22.39.3.61.22l2.49-1.01c.52.4 1.08.73 1.69.98l.38 2.65c.03.24.24.42.49.42h4c.25 0 .46-.18.49-.42l.38-2.65c.61-.25 1.17-.58 1.69-.98l2.49 1.01c.22.08.49 0 .61-.22l2-3.46c.12-.22.07-.49-.12-.64L19.43 12.97Z" fill="#555"/>
</svg>""",
}

# Add file-based samples
SAMPLE_DIR = Path(__file__).parent / ".." / "samples" / "test"


def _parse_viewbox(svg_code: str) -> tuple[float, float]:
    """Extract canvas dimensions from SVG viewBox attribute.

    Falls back to W3C CSS 2.1 default (300×150) if no viewBox found.
    This is not a magic number — it's the W3C standard for replaced elements
    with no intrinsic dimensions (CSS 2.1 Section 10.3.2).
    """
    m = re.search(r'viewBox\s*=\s*"([^"]+)"', svg_code)
    if m:
        parts = m.group(1).split()
        if len(parts) == 4:
            return float(parts[2]), float(parts[3])
    return 300.0, 150.0  # W3C default


def load_sample(name: str, path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  [skip] {name}: {e}")
        return None


def main():
    # Add file-based SVGs
    flink_path = SAMPLE_DIR / "apache-flink-icon (1).svg"
    faker_path = SAMPLE_DIR / "faker.svg"
    anthropic_path = SAMPLE_DIR / "anthropic.svg"

    for name, path in [("Flink Squirrel", flink_path), ("Faker Panda", faker_path), ("Anthropic", anthropic_path)]:
        svg = load_sample(name, path)
        if svg:
            SAMPLES[name] = svg

    output_dir = Path(__file__).parent
    all_results = []

    for name, svg_code in SAMPLES.items():
        print(f"\n{'='*64}")
        print(f"  Processing: {name}")
        print(f"{'='*64}")

        # Parse viewBox for data-derived canvas dimensions
        canvas_w, canvas_h = _parse_viewbox(svg_code)

        t0 = time.perf_counter()
        try:
            result = run_pixel_pipeline(
                svg_code=svg_code,
                n_elements=svg_code.count("<") // 2,  # rough estimate
                canvas_w=canvas_w,
                canvas_h=canvas_h,
            )
            elapsed = time.perf_counter() - t0

            # Format and print results
            text = format_result_text(result)
            enrichment = format_enrichment_sections(result)

            print(f"\n  Time: {elapsed*1000:.0f}ms")
            print(f"  Resolution: {result.resolution} (canvas {canvas_w}x{canvas_h})")
            print(text)
            print("\n--- ENRICHMENT OUTPUT ---")
            print(enrichment)

            all_results.append((name, result, text, enrichment, elapsed))

        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"\n  ERROR after {elapsed*1000:.0f}ms: {e}")
            import traceback
            traceback.print_exc()

    # Save all results to file
    out_path = output_dir / "pixel_seg_results.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for name, result, text, enrichment, elapsed in all_results:
            f.write(f"\n{'='*64}\n")
            f.write(f"  {name} ({elapsed*1000:.0f}ms)\n")
            f.write(f"{'='*64}\n")
            f.write(text)
            f.write("\n--- ENRICHMENT OUTPUT ---\n")
            f.write(enrichment)
            f.write("\n\n")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
