#!/usr/bin/env python3
"""Run local data perturber for Anthropic sycophancy dataset."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.local_data_perturber import main


if __name__ == "__main__":
    main()
