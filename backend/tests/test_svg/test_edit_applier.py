"""Tests for the surgical SVG edit applier."""

from __future__ import annotations

import pytest

from app.models.edit_ops import EditOp, EditPlan
from app.svg.edit_applier import apply_edits
from app.svg.parser import parse_svg
from tests.conftest import BAR_CHART_SVG, HOME_SVG, SMILEY_SVG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(svg: str):
    """Parse SVG and return (svg_raw, ctx)."""
    ctx = parse_svg(svg)
    return svg, ctx


# ---------------------------------------------------------------------------
# 1. Source tracking
# ---------------------------------------------------------------------------

class TestSourceTracking:
    def test_parser_source_tracking_circles(self):
        """Circles should have source_tag and source_span populated."""
        ctx = parse_svg(SMILEY_SVG)
        circles = [sp for sp in ctx.subpaths if sp.source_tag.startswith("<circle")]
        assert len(circles) >= 1
        for sp in circles:
            assert sp.source_tag.startswith("<circle")
            assert sp.source_span != (0, 0)
            # Verify span points to the actual tag in svg_raw
            start, end = sp.source_span
            assert ctx.svg_raw[start:end] == sp.source_tag

    def test_parser_source_tracking_paths(self):
        """Paths should have source_tag and source_span populated."""
        ctx = parse_svg(SMILEY_SVG)
        paths = [sp for sp in ctx.subpaths if sp.source_tag.startswith("<path")]
        assert len(paths) >= 1
        for sp in paths:
            assert sp.source_tag.startswith("<path")
            start, end = sp.source_span
            assert ctx.svg_raw[start:end] == sp.source_tag

    def test_parser_source_tracking_lines(self):
        """Lines should have source_tag and source_span populated."""
        ctx = parse_svg(BAR_CHART_SVG)
        lines = [sp for sp in ctx.subpaths if sp.source_tag.startswith("<line")]
        assert len(lines) >= 1
        for sp in lines:
            assert sp.source_tag.startswith("<line")
            start, end = sp.source_span
            assert ctx.svg_raw[start:end] == sp.source_tag


# ---------------------------------------------------------------------------
# 2-3. Delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_element(self):
        """Deleting an element removes its tag from the SVG."""
        svg, ctx = _parse(SMILEY_SVG)
        # Find a circle element (eye)
        target = next(sp for sp in ctx.subpaths if "r" in sp.attributes and sp.attributes.get("r") == "1")
        ops = [EditOp(action="delete", target=target.id)]
        result = apply_edits(svg, ops, ctx)
        assert target.source_tag not in result
        # Other elements should still be present
        remaining = [sp for sp in ctx.subpaths if sp.id != target.id]
        for sp in remaining:
            assert sp.source_tag in result


# ---------------------------------------------------------------------------
# 4-5. Add
# ---------------------------------------------------------------------------

class TestAdd:
    def test_add_element_after(self):
        """Adding after an element inserts the fragment right after that tag."""
        svg, ctx = _parse(SMILEY_SVG)
        first = ctx.subpaths[0]
        new_circle = '<circle cx="5" cy="5" r="2"/>'
        ops = [EditOp(action="add", position=f"after:{first.id}", svg_fragment=new_circle)]
        result = apply_edits(svg, ops, ctx)
        assert new_circle in result
        # The new tag should appear after the first element's source_tag
        idx_first_end = result.find(first.source_tag) + len(first.source_tag)
        idx_new = result.find(new_circle)
        assert idx_new > idx_first_end

    def test_add_element_end(self):
        """Adding at 'end' inserts before </svg>."""
        svg, ctx = _parse(SMILEY_SVG)
        new_rect = '<rect x="0" y="0" width="5" height="5"/>'
        ops = [EditOp(action="add", position="end", svg_fragment=new_rect)]
        result = apply_edits(svg, ops, ctx)
        assert new_rect in result
        idx_rect = result.find(new_rect)
        idx_close = result.find("</svg>")
        assert idx_rect < idx_close

    def test_add_element_default_position(self):
        """Adding with no position defaults to 'end'."""
        svg, ctx = _parse(SMILEY_SVG)
        new_line = '<line x1="0" y1="0" x2="10" y2="10"/>'
        ops = [EditOp(action="add", svg_fragment=new_line)]
        result = apply_edits(svg, ops, ctx)
        assert new_line in result


# ---------------------------------------------------------------------------
# 6-7. Modify
# ---------------------------------------------------------------------------

class TestModify:
    def test_modify_attributes(self):
        """Modifying an attribute changes only that attribute."""
        svg, ctx = _parse(SMILEY_SVG)
        # Find the big circle (r="10")
        target = next(sp for sp in ctx.subpaths if sp.attributes.get("r") == "10")
        ops = [EditOp(action="modify", target=target.id, attributes={"r": "15"})]
        result = apply_edits(svg, ops, ctx)
        assert 'r="15"' in result
        assert 'r="10"' not in result or result.count('r="10"') < svg.count('r="10"')

    def test_modify_preserves_untouched(self):
        """Modifying one element preserves other elements byte-identical."""
        svg, ctx = _parse(SMILEY_SVG)
        target = next(sp for sp in ctx.subpaths if sp.attributes.get("r") == "10")
        ops = [EditOp(action="modify", target=target.id, attributes={"r": "15"})]
        result = apply_edits(svg, ops, ctx)
        # Other elements should be byte-identical
        for sp in ctx.subpaths:
            if sp.id != target.id:
                assert sp.source_tag in result


# ---------------------------------------------------------------------------
# 8. Multiple operations
# ---------------------------------------------------------------------------

class TestMultipleOps:
    def test_multiple_ops(self):
        """Apply delete + add + modify in one pass."""
        svg, ctx = _parse(SMILEY_SVG)
        # Delete an eye (r=1 circle)
        eye = next(sp for sp in ctx.subpaths if sp.attributes.get("r") == "1")
        # Modify the face (r=10 circle)
        face = next(sp for sp in ctx.subpaths if sp.attributes.get("r") == "10")
        # Add a new element
        new_el = '<ellipse cx="12" cy="18" rx="4" ry="2"/>'

        ops = [
            EditOp(action="delete", target=eye.id),
            EditOp(action="modify", target=face.id, attributes={"r": "12"}),
            EditOp(action="add", position="end", svg_fragment=new_el),
        ]
        result = apply_edits(svg, ops, ctx)

        assert eye.source_tag not in result
        assert 'r="12"' in result
        assert new_el in result


# ---------------------------------------------------------------------------
# 9. Invalid target
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_invalid_target_skipped(self):
        """An op with a non-existent element ID is skipped gracefully."""
        svg, ctx = _parse(SMILEY_SVG)
        ops = [EditOp(action="delete", target="E999")]
        result = apply_edits(svg, ops, ctx)
        assert result == svg  # No changes

    def test_conflicting_ops_delete_wins(self):
        """Delete + modify on the same target: delete wins."""
        svg, ctx = _parse(SMILEY_SVG)
        target = next(sp for sp in ctx.subpaths if sp.attributes.get("r") == "10")
        ops = [
            EditOp(action="delete", target=target.id),
            EditOp(action="modify", target=target.id, attributes={"r": "20"}),
        ]
        result = apply_edits(svg, ops, ctx)
        # Element should be deleted, not modified
        assert target.source_tag not in result
        assert 'r="20"' not in result


# ---------------------------------------------------------------------------
# 10. EditPlan parsing
# ---------------------------------------------------------------------------

class TestEditPlanParsing:
    def test_edit_plan_from_json(self):
        """Parse sample JSON into an EditPlan model."""
        raw = {
            "reasoning": "The user wants to remove the smile and enlarge the face.",
            "operations": [
                {"action": "delete", "target": "E4"},
                {"action": "modify", "target": "E1", "attributes": {"r": "12"}},
                {"action": "add", "position": "end", "svg_fragment": '<circle cx="12" cy="16" r="1"/>'},
            ],
        }
        plan = EditPlan(**raw)
        assert plan.reasoning.startswith("The user")
        assert len(plan.operations) == 3
        assert plan.operations[0].action == "delete"
        assert plan.operations[1].action == "modify"
        assert plan.operations[2].action == "add"
        assert plan.operations[2].svg_fragment is not None
