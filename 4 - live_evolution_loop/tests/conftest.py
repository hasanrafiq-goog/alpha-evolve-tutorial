"""Test configuration and path setup for Step 4 test suite."""

import os
import sys

# Ensure Step 4 root is in sys.path
STEP4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if STEP4_DIR not in sys.path:
    sys.path.insert(0, STEP4_DIR)
