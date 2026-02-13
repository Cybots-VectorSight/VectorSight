"""Shared test fixtures."""

from __future__ import annotations

import pytest


# Sample SVGs from data_spec.md

CIRCLE_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="12" cy="12" r="10"/>
</svg>'''

SMILEY_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="12" cy="12" r="10"/>
  <circle cx="8" cy="9" r="1"/>
  <circle cx="16" cy="9" r="1"/>
  <path d="M8 14s1.5 2 4 2 4-2 4-2"/>
</svg>'''

HOME_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M15 21v-8a1 1 0 0 0-1-1h-4a1 1 0 0 0-1 1v8"/>
  <path d="M3 10a2 2 0 0 1 .709-1.528l7-5.999a2 2 0 0 1 2.582 0l7 5.999A2 2 0 0 1 21 10v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
</svg>'''

BAR_CHART_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <line x1="18" x2="18" y1="20" y2="10"/>
  <line x1="12" x2="12" y1="20" y2="4"/>
  <line x1="6" x2="6" y1="20" y2="14"/>
</svg>'''

SETTINGS_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/>
  <circle cx="12" cy="12" r="3"/>
</svg>'''


# Filled SVG for breakdown testing (has areas, not just strokes)
FILLED_RECT_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect x="10" y="10" width="80" height="80" fill="#4ECDC4"/>
  <circle cx="50" cy="50" r="20" fill="#FF6B6B"/>
</svg>'''

FILLED_COMPLEX_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 259">
  <path d="M128 10 L240 80 L240 200 L128 249 L16 200 L16 80 Z" fill="#4ECDC4"/>
  <path d="M128 50 L200 100 L200 180 L128 220 L56 180 L56 100 Z" fill="#45B7D1"/>
  <circle cx="128" cy="130" r="30" fill="#FF6B6B"/>
  <circle cx="100" cy="110" r="10" fill="#FFEAA7"/>
  <circle cx="156" cy="110" r="10" fill="#FFEAA7"/>
</svg>'''


@pytest.fixture
def circle_svg() -> str:
    return CIRCLE_SVG


@pytest.fixture
def smiley_svg() -> str:
    return SMILEY_SVG


@pytest.fixture
def home_svg() -> str:
    return HOME_SVG


@pytest.fixture
def settings_svg() -> str:
    return SETTINGS_SVG


@pytest.fixture
def filled_rect_svg() -> str:
    return FILLED_RECT_SVG


@pytest.fixture
def filled_complex_svg() -> str:
    return FILLED_COMPLEX_SVG
