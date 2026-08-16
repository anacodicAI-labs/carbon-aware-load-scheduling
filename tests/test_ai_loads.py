from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cals.ai_loads import (
    PAI_TASK_COLUMNS,
    aggregate_alibaba_tasks,
    alibaba_to_jobs,
    iter_pai_task_table,
    load_pai_task_table,
    parse_alibaba_trace,
    uncapped_alibaba_costs,
)
from cuad.scheduler.greedy import fifo_baseline, schedule


class AlibabaTaskTableTests(unittest.TestCase):
    def test_headerless_table_has_official_columns_and_unique_task_ids(self) -> None:
        rows = "\n".join(
            [
                "job-1,worker,1,Terminated,0,3600,100,1,100,MISC",
                "job-1,evaluator,1,Terminated,3600,7200,100,1,50,MISC",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pai_task_table.csv"
            path.write_text(rows)
            raw = load_pai_task_table(path)

        parsed = parse_alibaba_trace(raw, origin="2019-06-01")

        self.assertEqual(list(raw.columns), list(PAI_TASK_COLUMNS))
        self.assertEqual(len(parsed), 2)
        self.assertTrue(parsed["job_id"].is_unique)

    def test_parser_keeps_only_completed_tasks_by_default(self) -> None:
        raw = load_pai_task_table_from_rows(
            [
                "job-complete,worker,1,Terminated,0,3600,100,1,100,MISC",
                "job-failed,worker,1,Failed,0,3600,100,1,100,MISC",
                "job-running,worker,1,Running,0,3600,100,1,100,MISC",
            ]
        )

        parsed = parse_alibaba_trace(raw, origin="2019-06-01")

        self.assertEqual(parsed["job_id"].tolist(), ["job-complete"])

    def test_chunk_reader_keeps_the_official_columns(self) -> None:
        rows = [
            "job-1,worker,1,Terminated,0,3600,100,1,100,MISC",
            "job-2,worker,1,Terminated,3600,7200,100,1,50,MISC",
            "job-3,worker,1,Terminated,7200,10800,100,1,25,MISC",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pai_task_table.csv"
            path.write_text("\n".join(rows))
            chunks = list(iter_pai_task_table(path, chunksize=2))

        self.assertEqual([len(chunk) for chunk in chunks], [2, 1])
        self.assertEqual(list(chunks[0].columns), list(PAI_TASK_COLUMNS))

    def test_uncapped_aggregate_costs_match_the_scheduler(self) -> None:
        parsed = pd.DataFrame(
            {
                "job_id": ["one", "two", "three"],
                "submit": pd.to_datetime(
                    ["2019-06-01 00:00", "2019-06-01 00:00", "2019-06-01 01:00"], utc=True
                ),
                "duration_h": [1.0, 1.0, 2.0],
                "num_gpus": [1.0, 2.0, 1.0],
            }
        )
        carbon = pd.Series(
            [100.0, 300.0, 50.0, 200.0, 400.0],
            index=pd.date_range("2019-06-01", periods=5, freq="h", tz="UTC"),
        )
        jobs = alibaba_to_jobs(parsed, gpu_power_kw=0.4, flex_hours=2, min_kwh=0.0)
        fifo = fifo_baseline(jobs, carbon)
        optimal = schedule(jobs, carbon)

        costs = uncapped_alibaba_costs(
            aggregate_alibaba_tasks(parsed), carbon, gpu_power_kw=0.4, flex_hours=2
        )

        self.assertEqual(costs["comparable_tasks"], len(jobs))
        self.assertEqual(costs["unpriced_baseline_tasks"], 0)
        self.assertAlmostEqual(costs["baseline_gco2"], fifo["total_gco2"])
        self.assertAlmostEqual(costs["scheduled_gco2"], optimal["total_gco2"])


def load_pai_task_table_from_rows(rows: list[str]):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pai_task_table.csv"
        path.write_text("\n".join(rows))
        return load_pai_task_table(path)


if __name__ == "__main__":
    unittest.main()
