"""Create reproducible, balanced subsets of the CPC18 raw data.

Balance means that a subset:

* contains every participant the same number of times;
* contains every game the same number of times within its set;
* contains complete 25-trial participant/game histories; and
* therefore preserves the exact trial, block, feedback, set, and location
  balance of the source data.

For ``rawData_Comp_All.csv``, one balanced round selects one complete game for
each of 240 participants and four participants for each of 60 games.  A round
therefore has 240 * 25 = 6,000 rows.  The possible non-empty sizes are 6,000,
12,000, ..., 180,000 rows.

The observed choice (``B``) and participant attributes such as gender are not
used for sampling.  Balancing on them would change the population being
studied or select trials based on the prediction target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_config import DATA_DIR, RAW_DATA_PATH


TRIALS_PER_GAME = 25
PARTICIPANTS_PER_SET = 120
GAMES_PER_SET = 30
NUMBER_OF_SETS = 2
ROWS_PER_BALANCED_ROUND = (
    NUMBER_OF_SETS * PARTICIPANTS_PER_SET * TRIALS_PER_GAME
)
MAX_BALANCED_ROUNDS = GAMES_PER_SET
MIN_BALANCED_ROWS = ROWS_PER_BALANCED_ROUND
MAX_BALANCED_ROWS = ROWS_PER_BALANCED_ROUND * MAX_BALANCED_ROUNDS
POSSIBLE_BALANCED_SIZES = tuple(
    range(MIN_BALANCED_ROWS, MAX_BALANCED_ROWS + 1, ROWS_PER_BALANCED_ROUND)
)


def closest_balanced_size(requested_rows: int) -> int:
    """Return the closest feasible non-empty subset size.

    Requests outside the feasible range are clamped.  An exact midpoint is
    rounded down so the result does not unexpectedly exceed the request.
    """
    if isinstance(requested_rows, bool) or not isinstance(requested_rows, int):
        raise TypeError("requested_rows must be an integer")

    if requested_rows <= MIN_BALANCED_ROWS:
        return MIN_BALANCED_ROWS
    if requested_rows >= MAX_BALANCED_ROWS:
        return MAX_BALANCED_ROWS

    lower = (requested_rows // ROWS_PER_BALANCED_ROUND) * ROWS_PER_BALANCED_ROUND
    upper = lower + ROWS_PER_BALANCED_ROUND
    if requested_rows - lower <= upper - requested_rows:
        return lower
    return upper


def _validate_raw_data(data: pd.DataFrame) -> None:
    """Fail clearly if the input is not the complete CPC18 panel expected here."""
    required = {"SubjID", "Set", "GameID", "Trial"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    sets = data["Set"].drop_duplicates().tolist()
    if len(sets) != NUMBER_OF_SETS:
        raise ValueError(f"Expected {NUMBER_OF_SETS} sets, found {len(sets)}")

    for set_value, set_rows in data.groupby("Set", sort=False):
        participants = set_rows["SubjID"].nunique()
        games = set_rows["GameID"].nunique()
        if participants != PARTICIPANTS_PER_SET or games != GAMES_PER_SET:
            raise ValueError(
                f"Set {set_value!r} has {participants} participants and {games} "
                f"games; expected {PARTICIPANTS_PER_SET} and {GAMES_PER_SET}"
            )

        cell_sizes = set_rows.groupby(["SubjID", "GameID"], sort=False).size()
        expected_cells = PARTICIPANTS_PER_SET * GAMES_PER_SET
        if len(cell_sizes) != expected_cells or not cell_sizes.eq(TRIALS_PER_GAME).all():
            raise ValueError(
                f"Set {set_value!r} is not a complete participant/game panel "
                f"with {TRIALS_PER_GAME} rows per cell"
            )

        trial_counts = set_rows.groupby(["SubjID", "GameID"])["Trial"].nunique()
        if not trial_counts.eq(TRIALS_PER_GAME).all():
            raise ValueError(
                f"Set {set_value!r} has duplicate or missing trial numbers"
            )


def create_balanced_subset(
    requested_rows: int,
    *,
    input_csv: str | Path = RAW_DATA_PATH,
    output_csv: str | Path | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Create the closest balanced subset to ``requested_rows``.

    The same seed produces the same subset.  With a fixed seed, feasible
    subsets are nested: a 12,000-row subset contains the corresponding
    6,000-row subset.  If ``output_csv`` is supplied, the result is also saved.
    """
    chosen_rows = closest_balanced_size(requested_rows)
    rounds = chosen_rows // ROWS_PER_BALANCED_ROUND

    data = pd.read_csv(input_csv)
    _validate_raw_data(data)
    rng = np.random.default_rng(seed)
    chosen_cells: set[tuple[object, object, object]] = set()

    # In each round, every participant receives one new game.  Because there
    # are 120 participants and 30 games per set, every game receives exactly
    # four participants per round.
    for set_value, set_rows in data.groupby("Set", sort=False):
        participants = np.array(set_rows["SubjID"].drop_duplicates().tolist())
        games = np.array(set_rows["GameID"].drop_duplicates().tolist())
        rng.shuffle(participants)
        rng.shuffle(games)

        for participant_index, participant in enumerate(participants):
            for round_index in range(rounds):
                game = games[(participant_index + round_index) % len(games)]
                chosen_cells.add((set_value, participant, game))

    row_keys = pd.MultiIndex.from_frame(data[["Set", "SubjID", "GameID"]])
    selected_keys = pd.MultiIndex.from_tuples(
        chosen_cells, names=["Set", "SubjID", "GameID"]
    )
    subset = data.loc[row_keys.isin(selected_keys)].copy()

    if len(subset) != chosen_rows:
        raise RuntimeError(f"Expected {chosen_rows} selected rows, got {len(subset)}")

    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(output_path, index=False)

    return subset


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requested_rows", type=int)
    parser.add_argument("--input", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    selected_size = closest_balanced_size(args.requested_rows)
    output = args.output or DATA_DIR / f"balanced_{selected_size}.csv"
    subset = create_balanced_subset(
        args.requested_rows,
        input_csv=args.input,
        output_csv=output,
        seed=args.seed,
    )
    rounds = len(subset) // ROWS_PER_BALANCED_ROUND
    print(
        f"Wrote {len(subset):,} rows ({rounds} balanced round(s)) to {output}"
    )


if __name__ == "__main__":
    _main()
