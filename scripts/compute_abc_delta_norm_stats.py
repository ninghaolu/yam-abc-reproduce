#!/usr/bin/env python3
"""CLI wrapper for full global ABC/YAM delta-action normalization statistics."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from yam_abc_reproduce.data.abc_delta_norm_stats import main

    raise SystemExit(main())
