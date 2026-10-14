import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from check_warm_deep_paths import WarmContentCut


class WarmCutTest(unittest.TestCase):
    def test_each_independent_history_retains_its_own_prefix(self):
        cut=WarmContentCut(2,4);x=torch.tensor([[1.,2.]])
        for position in range(8):
            output=cut(None,(),x)
            self.assertTrue(torch.equal(output,x if position%4<2 else torch.zeros_like(x)))
        self.assertTrue(torch.equal(x,torch.tensor([[1.,2.]])))

    def test_invalid_empty_windows_are_rejected(self):
        with self.assertRaises(ValueError):WarmContentCut(4,4)
