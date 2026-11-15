"""Cognitive-model forecasts and simple reference baselines for CPC18.

The cognitive models are the public CPC18 competition forecasts:

* BEAST     -- the organizers' baseline model (file ``BEAST_sd.csv``);
* BEAST-GB  -- the competition winner (file ``BP47.csv``).

Each file has 60 rows, one per test problem in ascending GameID order, and five
columns, one per block of five trials. A value is the predicted rate of
choosing Option B. These forecasts depend only on the task; unlike the LLM
prompts they do not use the individual participant's earlier choices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FORECAST_FILES = {"BEAST": "BEAST_sd.csv", "BEAST-GB": "BP47.csv"}
BLOCKS = 5
KEYS = ["SubjID", "Set", "GameID"]


def load_forecasts(data_dir: Path, game_ids) -> pd.DataFrame:
    """Return available forecasts as rows of model, GameID, block, probability_B.

    ``game_ids`` are the test problems' GameIDs; forecast rows are matched to
    them in ascending order. Missing forecast files are skipped.
    """
    games = sorted(set(int(game) for game in game_ids))
    frames = []
    for model, filename in FORECAST_FILES.items():
        path = Path(data_dir) / filename
        if not path.is_file():
            continue
        values = pd.read_csv(path).to_numpy(dtype=float)
        if values.shape != (len(games), BLOCKS):
            raise ValueError(
                f"{filename} has shape {values.shape}; expected "
                f"({len(games)}, {BLOCKS}) to match the test problems"
            )
        if not ((values >= 0) & (values <= 1)).all():
            raise ValueError(f"{filename} contains values outside 0-1")
        frames.append(
            pd.DataFrame(
                {
                    "model": model,
                    "GameID": np.repeat(games, BLOCKS),
                    "block": np.tile(np.arange(1, BLOCKS + 1), len(games)),
                    "probability_B": values.ravel(),
                }
            )
        )
    columns = ["model", "GameID", "block", "probability_B"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def forecast_probabilities(data: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """Return one p(B) column per cognitive model, aligned with ``data``."""
    result = pd.DataFrame(index=data.index)
    for model, group in forecasts.groupby("model", sort=False):
        lookup = group.set_index(["GameID", "block"]).probability_B
        keys = pd.MultiIndex.from_frame(data[["GameID", "block"]].astype(int))
        result[model] = lookup.reindex(keys).to_numpy()
    return result


def reference_baselines(data: pd.DataFrame) -> pd.DataFrame:
    """Return simple p(B) baselines aligned with ``data``.

    ``data`` must contain complete trial sequences, because two baselines use
    the same earlier choices that the LLM prompts contain:

    * ``constant_0.5``     -- no information;
    * ``repeat_previous``  -- the participant's previous choice (0.5 on trial 1);
    * ``history_rate``     -- Laplace-smoothed rate of earlier B choices.
    """
    ordered = data.sort_values(KEYS + ["Trial"], kind="stable")
    grouped = ordered.groupby(KEYS, sort=False)
    earlier_trials = grouped.cumcount()
    earlier_b = grouped.B.cumsum() - ordered.B
    result = pd.DataFrame(
        {
            "constant_0.5": 0.5,
            "repeat_previous": grouped.B.shift().fillna(0.5),
            "history_rate": (earlier_b + 1) / (earlier_trials + 2),
        },
        index=ordered.index,
    )
    return result.reindex(data.index)
