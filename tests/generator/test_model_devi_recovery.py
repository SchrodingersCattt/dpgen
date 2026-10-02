import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dpgen.generator.run import (
    _normalize_model_devi_recovery,
    _read_model_devi_file,
    _recovery_task_report,
)

DUMP = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 5
0 5
0 5
ITEM: ATOMS id type x y z fx fy fz
1 1 0 0 0 0 0 0
2 1 1 0 0 0 0 0
"""


class TestModelDeviRecovery(unittest.TestCase):
    def test_policy_defaults_and_validation(self):
        policy = _normalize_model_devi_recovery(
            {"model_devi_recovery": {"enabled": True}}
        )
        self.assertTrue(policy["enabled"])
        self.assertEqual(policy["max_failed_tasks"], 0)
        with self.assertRaises(ValueError):
            _normalize_model_devi_recovery(
                {"model_devi_recovery": {"max_failed_ratio": 2}}
            )

    def test_task_report_salvages_matching_prefix(self):
        with tempfile.TemporaryDirectory() as root:
            task = Path(root) / "task.000.000000"
            (task / "traj").mkdir(parents=True)
            (task / "traj" / "0.lammpstrj").write_text(DUMP)
            np.savetxt(
                task / "model_devi.out",
                np.asarray([[0, 0.1, 0, 0, 0.2, 0, 0], [10, 0.1, 0, 0, 0.2, 0, 0]]),
            )
            report = _recovery_task_report(task, False, True)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["valid_steps"], [0])
            self.assertEqual(report["excluded_steps"], [10])

    def test_selection_reads_report_allow_list(self):
        with tempfile.TemporaryDirectory() as root:
            iteration = Path(root) / "iter.000000"
            task = iteration / "01.model_devi" / "task.000.000000"
            task.mkdir(parents=True)
            np.savetxt(
                task / "model_devi.out",
                np.asarray([[0, 0.1, 0, 0, 0.2, 0, 0], [10, 0.1, 0, 0, 0.2, 0, 0]]),
            )
            (iteration / "exploration_report.json").write_text(
                json.dumps({"tasks": [{"task": task.name, "valid_steps": [0]}]})
            )
            result = _read_model_devi_file(str(task))
            self.assertEqual(result.shape, (1, 7))
            self.assertEqual(int(result[0, 0]), 0)


if __name__ == "__main__":
    unittest.main()
