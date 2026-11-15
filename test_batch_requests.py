"""Offline checks of the experiment grid and the batch requests."""

import unittest

from cpc18_row_to_english import NO_HISTORY_TEXT
from experiment_config import ANSWER_FORMAT, EFFORTS, INSTRUCTIONS, MODELS, Condition, conditions
from openai_batch_requests import build_prediction_requests
from test_prompt import synthetic_sequence


def user_prompt(request):
    return request["body"]["messages"][1]["content"]


class GridTests(unittest.TestCase):
    def test_grid_is_the_full_product(self):
        grid = conditions()
        self.assertEqual(len(grid), len(MODELS) * len(EFFORTS) * len(INSTRUCTIONS))
        self.assertEqual(len({condition.id for condition in grid}), len(grid))

    def test_grid_can_be_restricted(self):
        grid = conditions(models=["small", "org/large:v2"], efforts=["low"], instructions=["minimal"])
        self.assertEqual([c.model for c in grid], ["small", "org/large:v2"])
        self.assertEqual(grid[1].id, "org-large-v2__low__minimal")
        with self.assertRaises(ValueError):
            conditions(instructions=["does_not_exist"])

    def test_every_instruction_ends_with_the_answer_format(self):
        for text in INSTRUCTIONS.values():
            self.assertTrue(text.endswith(ANSWER_FORMAT))


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_sequence()
        self.condition = Condition("some-model", "high", "predict_human")
        self.requests = build_prediction_requests(self.data, self.condition)

    def test_one_request_per_row(self):
        self.assertEqual(len(self.requests), len(self.data))
        self.assertEqual(
            [request["custom_id"] for request in self.requests],
            [f"row-{number}" for number in range(len(self.data))],
        )
        self.assertTrue(all(r["url"] == "/v1/chat/completions" for r in self.requests))

    def test_body_comes_from_the_condition(self):
        body = self.requests[0]["body"]
        self.assertEqual(body["model"], "some-model")
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertEqual(body["messages"][0],
                         {"role": "developer", "content": INSTRUCTIONS["predict_human"]})
        self.assertTrue(body["logprobs"])

    def test_every_prompt_has_the_entire_earlier_history(self):
        self.assertIn(NO_HISTORY_TEXT, user_prompt(self.requests[0]))
        for request, trial in zip(self.requests, self.data.Trial):
            prompt = user_prompt(request)
            for earlier in range(1, trial):
                self.assertIn(f"\n{earlier}: ", prompt)
            self.assertNotIn(f"\n{trial}: ", prompt)

    def test_conditions_share_the_same_user_prompts(self):
        other = build_prediction_requests(self.data, Condition("other-model", "none", "minimal"))
        self.assertEqual([user_prompt(r) for r in self.requests], [user_prompt(r) for r in other])

    def test_participants_and_tasks_are_kept_apart(self):
        import pandas as pd

        mixed = pd.concat(
            [synthetic_sequence(subject=1, trials=3), synthetic_sequence(subject=2, trials=3)],
            ignore_index=True,
        )
        requests = build_prediction_requests(mixed, self.condition)
        self.assertEqual(len(requests), 6)
        self.assertIn(NO_HISTORY_TEXT, user_prompt(requests[3]))


if __name__ == "__main__":
    unittest.main()
