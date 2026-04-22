#!/usr/bin/env python
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.commands import validate_pope_data_main


if __name__ == "__main__":
    validate_pope_data_main()
