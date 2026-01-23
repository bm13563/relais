"""Shared configuration for examples."""

import os
from pathlib import Path

# Paths
EXAMPLES_DIR = Path(__file__).parent
INSTRUCTIONS_DIR = EXAMPLES_DIR / "instructions"

# Database configuration - SQLite file in examples directory
DB_PATH = str(EXAMPLES_DIR / "pipeline.db")

# Logging
LOG_FILE = os.getenv("LOG_FILE", "pipeline.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
