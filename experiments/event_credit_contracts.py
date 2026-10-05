import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
raise SystemExit(pytest.main(['-q', 'tests/test_event_credit_sites.py']))
