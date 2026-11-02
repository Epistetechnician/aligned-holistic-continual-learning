import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "synthetic_gate_sweep", ROOT / "scripts" / "synthetic_gate_sweep.py"
)
SWEEP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SWEEP
SPEC.loader.exec_module(SWEEP)


class SyntheticGateSweepTests(unittest.TestCase):
    def test_detailed_sweep_reports_each_gate_failure(self):
        details = SWEEP._sweep_seeds_detailed(
            [f"seed-{n}" for n in range(16)], 1.0, 16
        )

        self.assertEqual(set(details[3]), set(SWEEP.GATE_NAMES))
        self.assertEqual(details[2], "sign_agreement")
        self.assertEqual(details[3]["sign_agreement"], 6)
        self.assertEqual(details[3]["shuffled_control"], 1)

    def test_compatibility_wrapper_keeps_three_value_result(self):
        result = SWEEP._sweep_seeds(["seed-0"], 1.0, 16)

        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[2], str)

    def test_documented_noise_table_matches_sweep_output(self):
        addendum = ROOT / "docs" / "synthetic-gate-fragility-addendum.md"
        section = addendum.read_text(encoding="utf-8").split(
            "## Which gate fails first as noise rises", 1
        )[1].split("\n## ", 1)[0]
        columns = ("r2", "r2_over_baseline", "sign_agreement", "shuffled_control")
        rows = []
        for line in section.splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) != 7 or not cells[0].replace(".", "", 1).isdigit():
                continue
            rows.append(cells)
        self.assertEqual(len(rows), 6)

        seeds = [f"seed-{n}" for n in range(16)]
        for cells in rows:
            noise = float(cells[0])
            passes, _, binding, failures = SWEEP._sweep_seeds_detailed(seeds, noise, 16)

            self.assertEqual(cells[1], f"{passes / len(seeds):.3f}", msg=f"noise {noise}")
            self.assertEqual(cells[2], binding, msg=f"noise {noise}")
            self.assertEqual(
                [int(value) for value in cells[3:]],
                [failures[column] for column in columns],
                msg=f"noise {noise}",
            )


if __name__ == "__main__":
    unittest.main()
