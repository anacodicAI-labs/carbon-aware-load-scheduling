from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import nb_utils


class MarginalCiProxyTests(unittest.TestCase):
    def test_uses_oil_generation_to_choose_the_hourly_proxy(self) -> None:
        first = pd.Timestamp("2019-01-01 00:00", tz="UTC")
        second = pd.Timestamp("2019-01-01 01:00", tz="UTC")
        mix = pd.DataFrame(
            {
                "timestamp": [first, first, second],
                "fueltype": ["NG", "OIL", "NG"],
                "gen_mwh": [100.0, 1.0, 100.0],
            }
        )

        proxy = nb_utils.marginal_ci_proxy(mix)

        self.assertEqual(proxy.loc[first], 650.0)
        self.assertEqual(proxy.loc[second], 490.0)

    def test_notebook_reports_average_minus_marginal_delta(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "04_sweeps.ipynb"
        notebook = json.loads(notebook_path.read_text())
        cell = next(cell for cell in notebook["cells"] if cell.get("id") == "ma-average-marginal-run")
        source = "".join(cell["source"])

        self.assertIn(
            '"delta (pp)": round(average_savings - marginal_savings, 2)',
            source,
        )


if __name__ == "__main__":
    unittest.main()
