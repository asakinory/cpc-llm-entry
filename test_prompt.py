"""Offline checks of the prompt builder. No data files or API calls needed.

    python -m unittest
"""

import unittest

import pandas as pd

from cpc18_row_to_english import (
    NO_HISTORY_TEXT,
    cpc18_row_to_english,
    get_cpc18_row_and_history,
)


def synthetic_sequence(subject=1, game=211, trials=25, choices=None) -> pd.DataFrame:
    """Return made-up CPC18-format rows for one participant and one task."""
    rows = []
    for trial in range(1, trials + 1):
        chose_b = (trial % 3 == 0) if choices is None else bool(choices[trial - 1])
        a_pay, b_pay = (4 if trial % 5 else 0), 3
        rows.append(
            {
                "SubjID": subject, "Location": "Technion", "Gender": "F", "Age": 24,
                "Set": 8, "Condition": "ByProb", "GameID": game,
                "Ha": 4, "pHa": 0.8, "La": 0, "LotShapeA": "-", "LotNumA": 1,
                "Hb": 3, "pHb": 1, "Lb": 3, "LotShapeB": "-", "LotNumB": 1,
                "Amb": 0, "Corr": 0, "Order": 2, "Trial": trial,
                "Button": "R" if chose_b else "L", "B": int(chose_b),
                "Payoff": b_pay if chose_b else a_pay,
                "Forgone": a_pay if chose_b else b_pay,
                "RT": 800 + trial, "Apay": a_pay, "Bpay": b_pay,
                "Feedback": int(trial >= 6), "block": (trial - 1) // 5 + 1,
            }
        )
    return pd.DataFrame(rows)


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_sequence()

    def test_target_labels_never_change_the_prompt(self):
        target = self.data.iloc[24].to_dict()
        changed = dict(target)
        for field in ["B", "Button", "Payoff", "Forgone", "RT", "Apay", "Bpay"]:
            changed[field] = 999999
        history = self.data.iloc[:24]
        self.assertEqual(
            cpc18_row_to_english(target, history_rows=history),
            cpc18_row_to_english(changed, history_rows=history),
        )

    def test_history_at_or_after_the_target_is_rejected(self):
        with self.assertRaises(ValueError):
            cpc18_row_to_english(self.data.iloc[20], history_rows=self.data.iloc[:22])
        with self.assertRaises(ValueError):
            cpc18_row_to_english(self.data.iloc[20], history_rows=self.data.iloc[:21])

    def test_history_of_another_task_is_rejected(self):
        other = synthetic_sequence(game=212)
        with self.assertRaises(ValueError):
            cpc18_row_to_english(self.data.iloc[10], history_rows=other.iloc[:10])

    def test_selected_history_is_every_earlier_trial(self):
        history, row = get_cpc18_row_and_history(self.data, 25)
        self.assertEqual(row["Trial"], 25)
        self.assertEqual(history["Trial"].tolist(), list(range(1, 25)))

        prompt = cpc18_row_to_english(row, history_rows=history)
        self.assertIn("3: B; no feedback.", prompt)
        self.assertIn("24: B → 3; A → 4.", prompt)
        self.assertTrue(prompt.rstrip().endswith("A or B."))

    def test_first_trial_has_no_history(self):
        history, row = get_cpc18_row_and_history(self.data, 1)
        self.assertTrue(history.empty)
        self.assertIn(NO_HISTORY_TEXT, cpc18_row_to_english(row, history_rows=history))


if __name__ == "__main__":
    unittest.main()
