"""Offline checks of response parsing, cognitive models and the analysis."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import analyze, score_rows
from cognitive_models import load_forecasts, reference_baselines
from openai_batch_collect import parse_response_body
from predict_request_openai import choice_log_likelihoods, request_cost
from test_prompt import synthetic_sequence

GAMES = list(range(211, 271))


def response_body(answer="B", p_answer=0.8, with_logprobs=True):
    other = "A" if answer == "B" else "B"
    logprobs = None
    if with_logprobs:
        logprobs = {"content": [{
            "token": answer, "logprob": math.log(p_answer),
            "top_logprobs": [{"token": answer, "logprob": math.log(p_answer)},
                             {"token": " " + other, "logprob": math.log((1 - p_answer) / 2)},
                             {"token": "Option", "logprob": math.log((1 - p_answer) / 2)}],
        }]}
    return {"model": "some-model-2026", "choices": [{"message": {"content": answer}, "logprobs": logprobs}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 10}}


class ResponseTests(unittest.TestCase):
    def test_log_likelihoods_are_normalized_over_a_and_b(self):
        log_a, log_b = choice_log_likelihoods(response_body("B", 0.8))
        self.assertAlmostEqual(math.exp(log_a) + math.exp(log_b), 1.0)
        self.assertAlmostEqual(math.exp(log_b), 0.8 / 0.9)

    def test_missing_logprobs_give_none(self):
        self.assertEqual(choice_log_likelihoods(response_body(with_logprobs=False)), (None, None))

    def test_parse_response_body(self):
        pricing = {"input": 1.0, "cached_input": 0.1, "cache_write": 1.0, "output": 10.0,
                   "long_context_threshold": 272_000, "batch_multiplier": 0.5}
        parsed = parse_response_body(response_body("A", 0.6), pricing)
        self.assertEqual((parsed["model_answer"], parsed["model_choice"]), ("A", 0))
        self.assertAlmostEqual(parsed["estimated_cost_usd"], 0.5 * (1000 * 1.0 + 10 * 10.0) / 1e6)
        self.assertIsNone(parse_response_body(response_body(), None)["estimated_cost_usd"])
        with self.assertRaises(ValueError):
            parse_response_body({"choices": [{"message": {"content": "Option A"}}]}, None)

    def test_cost_is_none_without_prices(self):
        self.assertIsNone(request_cost({"prompt_tokens": 5, "completion_tokens": 1}, None))


class CognitiveModelTests(unittest.TestCase):
    def write_forecast(self, directory, name, values):
        pd.DataFrame(values, columns=[f"B{i}" for i in range(1, 6)]).to_csv(
            Path(directory) / name, index=False)

    def test_forecast_rows_follow_ascending_game_id(self):
        values = np.arange(300).reshape(60, 5) / 1000
        with tempfile.TemporaryDirectory() as directory:
            self.write_forecast(directory, "BP47.csv", values)
            forecasts = load_forecasts(directory, reversed(GAMES))
        self.assertEqual(forecasts.model.unique().tolist(), ["BEAST-GB"])
        lookup = forecasts.set_index(["GameID", "block"]).probability_B
        self.assertAlmostEqual(lookup[(211, 1)], 0.000)
        self.assertAlmostEqual(lookup[(211, 5)], 0.004)
        self.assertAlmostEqual(lookup[(270, 5)], 0.299)

    def test_forecast_with_wrong_shape_or_range_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_forecast(directory, "BP47.csv", np.zeros((59, 5)))
            with self.assertRaises(ValueError):
                load_forecasts(directory, GAMES)
            self.write_forecast(directory, "BP47.csv", np.full((60, 5), 1.5))
            with self.assertRaises(ValueError):
                load_forecasts(directory, GAMES)

    def test_missing_forecast_files_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(load_forecasts(directory, GAMES).empty)

    def test_reference_baselines_use_only_earlier_choices(self):
        data = synthetic_sequence(trials=4, choices=[1, 1, 0, 1])
        baselines = reference_baselines(data.iloc[::-1])  # row order must not matter
        baselines = baselines.loc[data.index]
        self.assertEqual(baselines["repeat_previous"].tolist(), [0.5, 1, 1, 0])
        np.testing.assert_allclose(baselines["history_rate"], [1 / 2, 2 / 3, 3 / 4, 3 / 5])
        self.assertEqual(baselines["constant_0.5"].tolist(), [0.5] * 4)


class AnalysisTests(unittest.TestCase):
    def test_scores(self):
        scores = score_rows([1, 0, 1], [1.0, 0.5, 0.2])
        np.testing.assert_allclose(scores.Brier, [0, 0.25, 0.64])
        np.testing.assert_allclose(scores.accuracy, [1, 0.5, 0])
        self.assertAlmostEqual(scores.NLL[2], -math.log(0.2))

    def test_end_to_end_on_made_up_predictions(self):
        data = pd.concat(
            [synthetic_sequence(subject=subject, game=game)
             for subject in (1, 2, 3) for game in GAMES],
            ignore_index=True,
        )

        def predictions(name, effort, probability_b, failed_rows=()):
            frame = data.copy()
            frame["condition"], frame["model"] = name, "some-model"
            frame["effort"], frame["instruction"] = effort, "minimal"
            frame["model_choice"] = (probability_b > 0.5).astype(float)
            frame["log_likelihood_B"] = np.log(probability_b)
            frame["batch_error"] = None
            frame.loc[list(failed_rows), ["batch_error", "model_choice"]] = ["failed", None]
            return frame

        with tempfile.TemporaryDirectory() as directory:
            outputs, data_dir = Path(directory) / "outputs", Path(directory) / "data"
            data_dir.mkdir()
            pd.DataFrame(np.full((60, 5), 1 / 3)).to_csv(data_dir / "BP47.csv", index=False)
            perfect = np.where(data.B == 1, 0.999, 0.001)
            for name, effort, probability, failed in [
                ("perfect", "high", perfect, ()),
                ("uninformed", "low", np.full(len(data), 0.4), (0, 1)),
            ]:
                (outputs / name).mkdir(parents=True)
                predictions(name, effort, probability, failed).to_csv(
                    outputs / name / "predictions.csv", index=False)

            metrics = analyze(outputs, data_dir).set_index("predictor")
            analysis = outputs / "analysis"
            for name in ["metrics.csv", "paired_vs_cognitive.csv", "by_factor.csv",
                         "aggregate_mse.csv", "by_trial.csv", "results.png", "REPORT.md"]:
                self.assertTrue((analysis / name).is_file(), name)
            paired = pd.read_csv(analysis / "paired_vs_cognitive.csv")
            by_factor = pd.read_csv(analysis / "by_factor.csv")

        # Rows that failed in one condition are dropped for every predictor.
        self.assertTrue((metrics.n == len(data) - 2).all())
        self.assertEqual(set(metrics.index), {"perfect", "uninformed", "BEAST-GB",
                                              "constant_0.5", "repeat_previous", "history_rate"})
        self.assertLess(metrics.Brier["perfect"], 1e-5)
        self.assertEqual(metrics.accuracy["perfect"], 1.0)
        self.assertAlmostEqual(metrics.Brier["constant_0.5"], 0.25)
        observed = metrics.observed_B_rate["BEAST-GB"]
        self.assertAlmostEqual(metrics.Brier["BEAST-GB"],
                               observed * (2 / 3) ** 2 + (1 - observed) * (1 / 3) ** 2)
        row = paired[paired.LLM.eq("perfect")].iloc[0]
        self.assertAlmostEqual(row.Brier_difference,
                               metrics.Brier["perfect"] - metrics.Brier["BEAST-GB"])
        self.assertLess(row.CI_high, 0)
        self.assertEqual(set(by_factor[by_factor.factor.eq("effort")].level), {"high", "low"})


if __name__ == "__main__":
    unittest.main()
