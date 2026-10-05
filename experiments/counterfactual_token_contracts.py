"""Bounded future-write contracts, admitted through run_safe.sh."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
raise SystemExit(pytest.main(['-q','tests/test_counterfactual_token_episodes.py']))
