#!/usr/bin/env python3
"""
Quick launcher for the modern Flask UI
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Flask UI with proper environment."""
    project_dir = Path(__file__).parent
    
    print("🚀 Launching Modern AI Story Generator UI...")
    print("✨ Unlimited scenes, modern design, real-time progress")
    print("🌐 Will open at: http://localhost:8080")
    print()
    
    # Change to project directory
    import os
    os.chdir(project_dir)

    # Prefer the currently running interpreter when already inside a venv (.venv/venv/uv)
    in_venv = bool(os.environ.get("VIRTUAL_ENV")) or (
        getattr(sys, "base_prefix", sys.prefix) != sys.prefix
    )

    candidates = []
    if in_venv:
        candidates.append(Path(sys.executable))

    # Common virtualenv locations (uv uses .venv by default)
    candidates.extend(
        [
            project_dir / ".venv" / "bin" / "python",
            project_dir / "venv" / "bin" / "python",
            Path(sys.executable),
        ]
    )

    python_exe = next((p for p in candidates if p and p.exists()), None)

    if not python_exe:
        print("❌ 找不到可用的 Python 解释器。")
        print("❌ Could not find a usable Python interpreter.")
        return

    print(f"🐍 Using python: {python_exe}")
    subprocess.run([str(python_exe), "flask_ui.py"])

if __name__ == "__main__":
    main()
