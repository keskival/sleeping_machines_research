"""Integrated token contracts; run only through the bounded queue runner."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
raise SystemExit(pytest.main(['-q', 'tests/test_integrated_token_core.py']))
