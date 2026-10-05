import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
raise SystemExit(pytest.main(['-q','tests/test_token_credit_boundaries.py']))
