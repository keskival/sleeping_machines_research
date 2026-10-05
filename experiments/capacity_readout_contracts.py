import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
raise SystemExit(pytest.main(['-q', 'tests/test_capacity_token_readout.py']))
