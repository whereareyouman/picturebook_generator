#!/usr/bin/env python3
"""
Compatibility wrapper (root-level).

This repository contains both:
- `gemini_picturebook_generator/` (the actual installable package)
- some root-level scripts kept for backwards compatibility / quick running

To avoid duplicated logic, the root-level `enhanced_story_generator.py` is now a thin
wrapper that re-exports the package implementation.
"""

from gemini_picturebook_generator.enhanced_story_generator import *  # noqa: F403


def main():  # pragma: no cover
    from gemini_picturebook_generator.enhanced_story_generator import main as _main

    _main()


if __name__ == "__main__":  # pragma: no cover
    main()
