import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import continue_market_depth_v2 as owner


class HandoffTest(unittest.TestCase):
    def test_market_failure_does_not_launch_withdrawn_language_and_signals_only_coordinator(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);tokens=root/'experiments/results/token_language';tokens.mkdir(parents=True)
            (root/'experiments/results/market').mkdir()
            (tokens/f'{owner.D16}.json').write_text(json.dumps({'status':'completed'}))
            def load(tag,kind='token_language'):
                return dict(status='completed',train_wall_s_total=1,presentations_total=32768)
            with patch.object(owner,'ROOT',root),patch.object(Path,'read_bytes',return_value=b'python continue_curie_deep_recovery_v2.py'),\
                 patch.object(owner.os,'kill') as kill,\
                 patch.object(owner.subprocess,'run',return_value=SimpleNamespace(returncode=0)),\
                 patch.object(owner.original,'run') as run,patch.object(owner.original,'load',side_effect=load),\
                 patch.object(owner,'stable_host'),patch.object(owner,'publish'),\
                 patch.object(owner,'market_stage',side_effect=RuntimeError('resource gate failed')),\
                 patch('sys.argv',['owner','--previous-controller-pid','123']),patch('builtins.print'):
                owner.main()
            self.assertEqual([c.args for c in kill.call_args_list],[(123,signal.SIGSTOP),(123,signal.SIGKILL)])
            self.assertEqual([c.args[0] for c in run.call_args_list],[owner.D16+'_quality'])
            self.assertTrue((root/f'experiments/results/market/{owner.FIT}_failure.json').exists())
