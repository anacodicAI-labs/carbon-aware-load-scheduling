from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import nb_utils


class BootstrapSavingsTests(unittest.TestCase):
    def test_daily_block_bootstrap_is_reproducible_and_covers_observed_savings(self) -> None:
        index = pd.date_range("2019-01-01", periods=5, freq="D", tz="UTC")
        baseline = pd.Series([100.0, 120.0, 140.0, 160.0, 200.0], index=index)
        scheduled = pd.Series([90.0, 96.0, 126.0, 120.0, 170.0], index=index)

        lower, upper = nb_utils.bootstrap_savings_ci(baseline, scheduled, n_boot=1_000, seed=42)
        repeat_lower, repeat_upper = nb_utils.bootstrap_savings_ci(
            baseline, scheduled, n_boot=1_000, seed=42
        )
        observed = 100.0 * (baseline.sum() - scheduled.sum()) / baseline.sum()

        self.assertEqual((lower, upper), (repeat_lower, repeat_upper))
        self.assertLessEqual(lower, observed)
        self.assertLessEqual(observed, upper)


if __name__ == "__main__":
    unittest.main()
