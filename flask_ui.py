#!/usr/bin/env python3
"""
Compatibility wrapper (root-level).

Why are there two `flask_ui.py` / `enhanced_story_generator.py` in this repo?
- `gemini_picturebook_generator/` is the actual installable package (single source of truth).
- Root-level scripts are kept for quick local runs / backwards compatibility.

To avoid duplicated maintenance, this root-level `flask_ui.py` delegates to the
package implementation: `gemini_picturebook_generator.flask_ui`.
"""

from gemini_picturebook_generator.flask_ui import app, main  # noqa: F401


if __name__ == "__main__":  # pragma: no cover
    main()


