from __future__ import annotations

import os
import sys
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from cals.acn_sessions import _token

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import nb_utils


class AcnSessionTests(unittest.TestCase):
    def test_token_reads_the_environment(self) -> None:
        with patch.dict(os.environ, {"ACN_API_TOKEN": "test-token"}, clear=True):
            self.assertEqual(_token(None), "test-token")

    @patch("cals.acn_sessions.fetch_acn_sessions", return_value=[{"sessionID": "session"}])
    @patch.object(nb_utils, "acn_to_jobs", return_value=["job"])
    def test_ev_jobs_fetches_2019_sessions(self, mock_jobs, mock_fetch) -> None:
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            with patch.object(nb_utils, "DATA", data_dir):
                jobs, source = nb_utils.get_ev_jobs(verbose=False)

        self.assertEqual(jobs, ["job"])
        self.assertEqual(source, "ACN-Data caltech 2019 (LIVE API)")
        mock_fetch.assert_called_once_with(
            date(2019, 1, 1),
            date(2019, 12, 31),
            cache_dir=data_dir / "ev",
        )


if __name__ == "__main__":
    unittest.main()
