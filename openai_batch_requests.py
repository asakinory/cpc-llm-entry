"""Build OpenAI Batch request records from CPC18 rows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from cpc18_row_to_english import cpc18_row_to_english
from experiment_config import Condition
from predict_request_openai import request_body


def build_prediction_requests(
    data: pd.DataFrame, condition: Condition
) -> list[dict[str, Any]]:
    """Return one Batch API request per row for one condition.

    Every prompt contains all earlier trials of the same participant and task.
    """
    required = {"SubjID", "Set", "GameID", "Trial"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    requests: list[dict[str, Any]] = []

    for _, group in data.groupby(["SubjID", "Set", "GameID"], sort=False):
        history: list[dict[str, Any]] = []

        for row_number, row in group.sort_values("Trial", kind="stable").iterrows():
            prompt = cpc18_row_to_english(row, history_rows=history)
            requests.append(
                {
                    "custom_id": f"row-{row_number}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": request_body(prompt, condition),
                }
            )
            history.append(row.to_dict())

    return requests
