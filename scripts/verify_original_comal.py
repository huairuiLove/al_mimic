#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mimic_comal.integrity import assert_original_unchanged


if __name__ == "__main__":
    print(json.dumps(assert_original_unchanged(), indent=2, allow_nan=False))
