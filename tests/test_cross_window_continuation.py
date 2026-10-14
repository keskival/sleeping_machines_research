"""Exercise continuation against the driver's real partial-result convention."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import continue_cross_window_language as controller


class ContinuationTest(unittest.TestCase):
    def test_checkpointed_smoke_partial_packet_admits_exact_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            tokens=root/'experiments/results/token_language';tokens.mkdir(parents=True)
            diagnostics=root/'experiments/results/diagnostics';diagnostics.mkdir()
            smoke=dict(status='checkpointed',max_rss_kb=1000,
                       train_wall_s_total=10,presentations_total=32768,process_wall_s=12,
                       curve=[dict(dev_nll=7,receiver_gradient_norms={'banks.input':[[[1.]] for _ in range(8)]})])
            (tokens/f'{controller.SMOKE}.partial.json').write_text(json.dumps(smoke))
            candidate=dict(status='completed',process_wall_s=30,credit_update_diagnostics={'steps':16384})
            source_diag=dict(status='completed',process_work_precision='recorded_process_intervals',
                             forward_horizon_contract={'status':'passed','maximum_after_boundary_feature_change':1.},
                             long_history_dev={'nll':5.},selected_long_history_dev={'nll':4.9})
            def load(tag,kind='token_language'):
                if kind=='diagnostics':return dict(source_diag)
                if tag==controller.FIT:return dict(candidate)
                return {'status':'completed'}
            with patch.object(controller,'ROOT',root),patch.object(controller,'stable_host'),\
                 patch.object(controller.original,'load',side_effect=load),\
                 patch.object(controller.original,'run') as run,\
                 patch.object(controller.subprocess,'run',return_value=SimpleNamespace(returncode=1)),\
                 patch('builtins.print'):
                controller.main()
            self.assertEqual([call.args[0] for call in run.call_args_list],
                             [controller.SMOKE,controller.FIT,controller.FIT+'_quality'])
            saved=json.loads((tokens/f'{controller.FIT}.json').read_text())
            self.assertEqual(saved['whole_fit_process_wall_s'],42)
            self.assertTrue((diagnostics/f'{controller.FIT}_decision.json').exists())
