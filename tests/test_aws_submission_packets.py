import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.tpp.submit import write_packet


class PacketTests(unittest.TestCase):
    def test_same_second_and_nanosecond_collision_preserves_first_packet(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('experiments.tpp.submit.time.time_ns', side_effect=[1, 1, 2]):
                first=write_packet(Path(folder),'R1','2','fixed',{'job':'fit'})
                second=write_packet(Path(folder),'R1','2','fixed',{'job':'analysis'})
            self.assertNotEqual(first,second)
            self.assertEqual(json.loads(first.read_text()),{'job':'fit'})
            self.assertEqual(json.loads(second.read_text()),{'job':'analysis'})


if __name__=='__main__': unittest.main()
