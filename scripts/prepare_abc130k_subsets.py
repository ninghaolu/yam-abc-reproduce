#!/usr/bin/env python3
"""Generate nested ABC-130K manifests and convert a cumulative LeRobot prefix."""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("yam_abc_reproduce.data.abc130k_subset", run_name="__main__")
