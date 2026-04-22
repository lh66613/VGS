#!/usr/bin/env python
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.commands import chair_sanity_check_main


if __name__ == "__main__":
    chair_sanity_check_main()
