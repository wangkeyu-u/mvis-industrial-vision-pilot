import unittest

from src.inference.performance import PerformanceSampler, unavailable_performance


class SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


class PerformanceTests(unittest.TestCase):
    def test_sampler_reports_warmup_percentiles_and_peak_memory(self) -> None:
        calls: list[str] = []
        memory = iter([100.0, 110.0, 105.0])
        sampler = PerformanceSampler(
            clock=SequenceClock([0.0, 0.010, 1.0, 1.020, 2.0, 2.040]),
            memory_reader=lambda: next(memory),
        )

        summary = sampler.run(
            lambda: calls.append("run"), warmup_runs=1, measured_runs=3
        )

        self.assertEqual(calls, ["run", "run", "run", "run"])
        self.assertTrue(summary.available)
        self.assertEqual(summary.warmup_runs, 1)
        self.assertEqual(summary.measured_runs, 3)
        self.assertEqual(summary.p50_latency_ms, 20.0)
        self.assertEqual(summary.p95_latency_ms, 40.0)
        self.assertEqual(summary.peak_memory_mb, 110.0)
        self.assertEqual(len(summary.samples), 3)

    def test_unavailable_summary_never_fabricates_measurements(self) -> None:
        summary = unavailable_performance(
            "real_model_unavailable", warmup_runs=1, requested_runs=5
        )

        self.assertFalse(summary.available)
        self.assertEqual(summary.measured_runs, 0)
        self.assertIsNone(summary.p50_latency_ms)
        self.assertIsNone(summary.p95_latency_ms)
        self.assertIsNone(summary.peak_memory_mb)
        self.assertEqual(summary.samples, ())

    def test_sampler_rejects_invalid_run_counts(self) -> None:
        sampler = PerformanceSampler()
        with self.assertRaisesRegex(ValueError, "warmup_runs"):
            sampler.run(lambda: None, warmup_runs=-1)
        with self.assertRaisesRegex(ValueError, "measured_runs"):
            sampler.run(lambda: None, measured_runs=0)


if __name__ == "__main__":
    unittest.main()
