"""Learning Memory — JSONL-based session store + pattern retrieval.

The chatbot accumulates knowledge from analysis sessions:
- Each session records: SVG features, prediction, actual answer, learnings
- Patterns are higher-level reusable insights derived from sessions
- On new analysis, relevant past learnings are injected into the prompt

NOT fine-tuning — this is structured experience memory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default data directory
_DEFAULT_DATA_DIR = Path(__file__).parent / "data"


@dataclass
class AnalysisSession:
    """Record of a single SVG analysis session."""

    svg_hash: str
    timestamp: float = 0.0
    element_count: int = 0
    fill_pct: float = 0.0
    symmetry_score: float = 0.0
    symmetry_axis: str = "none"
    composition_type: str = ""
    pose: str = ""
    shape_distribution: dict[str, int] = field(default_factory=dict)
    cluster_count: int = 0
    max_nesting_depth: int = 0
    # What was predicted vs what it actually was
    prediction: str = ""
    actual: str = ""
    # Derived learnings
    learnings: list[str] = field(default_factory=list)


@dataclass
class Pattern:
    """A reusable insight derived from analysis sessions."""

    id: str
    condition: str  # Human-readable condition description
    insight: str  # What this pattern teaches
    confidence: float = 0.5  # 0-1 confidence score
    times_confirmed: int = 0
    times_contradicted: int = 0
    tags: list[str] = field(default_factory=list)  # Feature tags for matching
    source_svg_hash: str = ""  # Which SVG this pattern was derived from


class MemoryStore:
    """JSONL-based learning memory for VectorSight."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or _DEFAULT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_file = self.data_dir / "sessions.jsonl"
        self.patterns_file = self.data_dir / "patterns.json"
        self._patterns_cache: list[Pattern] | None = None

    @staticmethod
    def svg_hash(svg: str) -> str:
        """Compute a short hash of SVG content for deduplication."""
        return hashlib.sha256(svg.encode()).hexdigest()[:12]

    def record_session(self, session: AnalysisSession) -> None:
        """Append a completed analysis session."""
        if not session.timestamp:
            session.timestamp = time.time()
        with open(self.sessions_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(session), ensure_ascii=False) + "\n")
        logger.info("Recorded session for SVG %s", session.svg_hash)

    def record_feedback(
        self,
        svg_hash: str,
        prediction: str,
        actual: str,
        learnings: list[str] | None = None,
    ) -> AnalysisSession | None:
        """Record user feedback on a prediction. Returns the updated session."""
        sessions = self._load_sessions()

        # Find the most recent session for this SVG
        target = None
        for s in reversed(sessions):
            if s.svg_hash == svg_hash:
                target = s
                break

        if target is None:
            # Create a minimal session
            target = AnalysisSession(
                svg_hash=svg_hash,
                timestamp=time.time(),
                prediction=prediction,
                actual=actual,
                learnings=learnings or [],
            )
            self.record_session(target)
            return target

        # Update the session
        target.prediction = prediction
        target.actual = actual
        if learnings:
            target.learnings.extend(learnings)

        # Rewrite sessions file
        self._save_sessions(sessions)

        # Auto-derive patterns from feedback
        if prediction != actual and learnings:
            self._add_pattern_from_feedback(target)

        return target

    def get_sessions(self, limit: int = 50) -> list[AnalysisSession]:
        """Get recent sessions."""
        sessions = self._load_sessions()
        return sessions[-limit:]

    def get_patterns(self) -> list[Pattern]:
        """Get all learned patterns."""
        if self._patterns_cache is not None:
            return self._patterns_cache
        self._patterns_cache = self._load_patterns()
        return self._patterns_cache

    def get_relevant_learnings(
        self,
        element_count: int = 0,
        symmetry_score: float = 0.0,
        fill_pct: float = 0.0,
        composition_type: str = "",
        shape_distribution: dict[str, int] | None = None,
        max_results: int = 5,
    ) -> list[str]:
        """Find patterns relevant to the current SVG's features."""
        patterns = self.get_patterns()
        if not patterns:
            return []

        scored: list[tuple[float, str]] = []

        # Factual tags only — nuanced tagging is the LLM's job
        from app.learning.tags import build_factual_tags

        tags = build_factual_tags(
            shape_distribution=shape_distribution,
            composition_type=composition_type,
        )

        # Score each pattern by tag overlap
        for pattern in patterns:
            if pattern.confidence < 0.3:
                continue
            overlap = len(tags.intersection(set(pattern.tags)))
            if overlap > 0:
                score = overlap * pattern.confidence
                scored.append((score, pattern.insight))

        # Sort by score and return top insights
        scored.sort(reverse=True)
        return [insight for _, insight in scored[:max_results]]

    def add_pattern(self, pattern: Pattern) -> None:
        """Add or update a pattern."""
        patterns = self._load_patterns()

        # Check for existing pattern with same ID
        for i, existing in enumerate(patterns):
            if existing.id == pattern.id:
                patterns[i] = pattern
                break
        else:
            patterns.append(pattern)

        self._save_patterns(patterns)
        self._patterns_cache = patterns

    def _add_pattern_from_feedback(self, session: AnalysisSession) -> None:
        """Auto-derive a pattern from a feedback session."""
        patterns = self._load_patterns()
        next_id = f"p{len(patterns) + 1:03d}"

        from app.learning.tags import build_factual_tags

        tags = list(build_factual_tags(
            shape_distribution=session.shape_distribution,
            composition_type=session.composition_type,
        ))

        for learning in session.learnings:
            pattern = Pattern(
                id=next_id,
                condition=f"Derived from {session.actual} (predicted {session.prediction})",
                insight=learning,
                confidence=0.7,
                times_confirmed=1,
                tags=tags,
            )
            patterns.append(pattern)
            next_id = f"p{len(patterns) + 1:03d}"

        self._save_patterns(patterns)
        self._patterns_cache = patterns

    def _load_sessions(self) -> list[AnalysisSession]:
        if not self.sessions_file.exists():
            return []
        sessions = []
        with open(self.sessions_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    sessions.append(AnalysisSession(**data))
        return sessions

    def _save_sessions(self, sessions: list[AnalysisSession]) -> None:
        with open(self.sessions_file, "w", encoding="utf-8") as f:
            for s in sessions:
                f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

    def _load_patterns(self) -> list[Pattern]:
        if not self.patterns_file.exists():
            return self._load_seed_patterns()
        with open(self.patterns_file, encoding="utf-8") as f:
            data = json.load(f)
        return [Pattern(**p) for p in data]

    def _load_seed_patterns(self) -> list[Pattern]:
        """Load seed patterns on first run."""
        seed_file = Path(__file__).parent / "seed_patterns.json"
        if not seed_file.exists():
            return []
        with open(seed_file, encoding="utf-8") as f:
            data = json.load(f)
        patterns = [Pattern(**p) for p in data]
        # Save as initial patterns
        self._save_patterns(patterns)
        return patterns

    def _save_patterns(self, patterns: list[Pattern]) -> None:
        with open(self.patterns_file, "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in patterns], f, indent=2, ensure_ascii=False)


# Singleton
_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    """Get or create the global MemoryStore singleton."""
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


def record_from_context(
    ctx: Any,
    svg: str,
    question: str = "",
    answer: str = "",
) -> None:
    """Auto-record an analysis session from a PipelineContext.

    Called after every chat/modify interaction so the system learns
    from every user — what SVGs are being analyzed, what features
    they have, and what questions are asked.
    """
    try:
        store = get_memory_store()
        svg_h = store.svg_hash(svg)

        # Extract features from context
        shape_dist: dict[str, int] = {}
        for sp in ctx.subpaths:
            s = sp.features.get("shape_class", "organic")
            shape_dist[s] = shape_dist.get(s, 0) + 1

        fill_pct = 0.0
        if ctx.subpaths:
            fill_pct = ctx.subpaths[0].features.get("positive_fill_pct", 0.0)

        max_depth = 0
        if ctx.containment_matrix is not None:
            n = len(ctx.subpaths)
            for i in range(n):
                depth = sum(
                    1 for j in range(n)
                    if j != i and ctx.containment_matrix[j][i]
                )
                max_depth = max(max_depth, depth)

        cluster_count = 0
        if ctx.cluster_labels is not None:
            cluster_count = len(set(int(l) for l in ctx.cluster_labels if l >= 0))

        # Build composition type from interpreter
        from app.engine.interpreter import _interpret_composition, _interpret_orientation
        comp = _interpret_composition(ctx)
        pose = _interpret_orientation(ctx, [])

        session = AnalysisSession(
            svg_hash=svg_h,
            element_count=len(ctx.subpaths),
            fill_pct=fill_pct,
            symmetry_score=ctx.symmetry_score,
            symmetry_axis=ctx.symmetry_axis or "none",
            composition_type=comp,
            pose=pose,
            shape_distribution=shape_dist,
            cluster_count=cluster_count,
            max_nesting_depth=max_depth,
            prediction=question,  # Store what was asked
            actual=answer[:200] if answer else "",  # Store truncated response
        )
        store.record_session(session)
        logger.debug("Auto-recorded session for SVG %s (%d elements)", svg_h, len(ctx.subpaths))
    except Exception as e:
        logger.warning("Failed to auto-record session: %s", e)
