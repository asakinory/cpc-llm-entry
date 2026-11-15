"""Turn CPC18 raw-data rows into plain-English prompts for LLM choice prediction.

The main entry point is :func:`cpc18_row_to_english`. It accepts the target
trial row plus the earlier rows for the same participant and task,
and returns one prompt asking which option the participant chose on the target
trial. The target row's choice and post-choice data are never included.

CPC18 column glossary (Plonsky et al., 2025, Nature Human Behaviour, Table 1)
-----------------------------------------------------------------------------
Ha, pHa, La     Option A pays Ha with probability pHa, otherwise La. When the
                option has multiple high outcomes, Ha is the expected value of
                the high-outcome lottery.
LotShapeA       Shape of A's high-outcome lottery: 'Symm', 'R-Skew', 'L-Skew',
                or '-' when A has a single fixed high outcome.
LotNumA         Number of outcomes in A's high-outcome lottery (1 if none).
Hb, pHb, Lb, LotShapeB, LotNumB
                Same five dimensions for Option B.
Amb             1 if Option B's probabilities were hidden from the participant
                (choice under ambiguity), 0 if they were shown.
Corr            Sign of the correlation between the payoffs drawn for the two
                options on each trial (-1, 0, or 1). Not shown to participants.
block           Block number; each block is 5 of the 25 trials.
Feedback        1 if the trial showed feedback (trials 6-25), 0 otherwise.
Trial, Order    Trial number within the task (1-25) and the task's position in
                the participant's session (each participant played 30 tasks).
SubjID, Location, Gender, Age, Set, Condition, GameID
                Participant and bookkeeping fields.
B, Button, Payoff, Forgone, RT, Apay, Bpay
                The participant's choice and its outcomes -- the prediction
                label and post-choice data, never written into the prompt.

Prompt-design principles (why the text looks the way it does)
-------------------------------------------------------------
1.  Describe the task as the participant saw it. Each option's full outcome
    distribution is spelled out, mirroring the CPC18 choice screen. When
    ``Amb == 1``, the participant could not see Option B's probabilities, so
    the prompt first shows only B's possible outcomes and then gives the true
    distribution explicitly labelled as hidden from the participant (the model
    needs it to reason about the feedback the participant received).
2.  No jargon or raw codes. Every parameter is translated into plain English;
    codes such as ``Corr=-1`` or ``R-Skew`` never appear in the prompt. The
    translations live in the "TRANSLATION TABLES" section below, so changing
    the wording never requires touching the logic.
3.  Stable structure, question last. Sections always appear in the same order
    (instructions -> participant -> task -> history -> question), the question
    is the final line, and it states the exact answer format ("A or B"). This
    keeps the request unambiguous and easy for an LLM to follow.
4.  No leakage. The target trial's choice, payoffs, and response time are never
    read into the prompt, and history contains only earlier trials. Feedback
    payoffs appear in history only for trials where feedback was shown.
5.  Compact history. One short line per earlier trial, with the legend given
    once, keeps long histories cheap in tokens without losing information.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


REQUIRED_COLUMNS = {
    "SubjID", "Location", "Gender", "Age", "Set", "Condition", "GameID",
    "Ha", "pHa", "La", "LotShapeA", "LotNumA", "Hb", "pHb", "Lb",
    "LotShapeB", "LotNumB", "Amb", "Corr", "Order", "Trial", "Button",
    "B", "Payoff", "Forgone", "RT", "Apay", "Bpay", "Feedback", "block",
}


# =============================================================================
# TRANSLATION TABLES -- edit here to change the English wording of the prompt.
#
# Every piece of prose in the prompt comes from this section. The rest of the
# module only parses values and fills these templates in.
# =============================================================================

# Opening paragraph explaining the CPC18 procedure (Plonsky et al., 2025, Fig 1).
GENERAL_INTRODUCTION = (
    "In this experiment, a participant repeatedly chose between two options, "
    "A and B, for 25 consecutive trials of the same task. On trials 1-5 they "
    "received no feedback about outcomes. From trial 6 onward, after every "
    "choice they saw both the points they received and the points the other "
    "option would have paid."
)

# Participant sentence. {gender_age} is e.g. "female, age 23" or "age 23".
PARTICIPANT_TEMPLATE = "The participant was {gender_age}, tested in {location}."
GENDER_ENGLISH = {"f": "female", "m": "male"}  # anything else: gender omitted

# Task context. {order} is an ordinal like "9th".
TASK_CONTEXT_TEMPLATE = (
    "This was the {order} of 30 tasks in the participant's session."
)
# Shown for the 'ByProb' condition (the only condition in the raw data):
CONDITION_BY_PROB = "All 25 trials of a task were completed before the next task."
CONDITION_OTHER_TEMPLATE = "Condition: {condition}."

# Option descriptions. Outcomes are listed from highest to lowest, matching
# the CPC18 choice screen. {outcome} already includes the unit ("5 points").
OPTION_SURE_TEMPLATE = "Option {label} always paid {outcome}."
OPTION_RISKY_TEMPLATE = "Option {label} paid {distribution}."
OUTCOME_TEMPLATE = "{outcome} with {probability} chance"
# When a hand summary of a multi-outcome lottery helps the reader:
LOTTERY_SHAPE_TEMPLATE = (
    "Option {label}'s {count} high outcomes came from a {shape} lottery with "
    "mean {mean} points."
)
LOT_SHAPE_ENGLISH = {
    "symm": "symmetric",
    "r-skew": "right-skewed",
    "l-skew": "left-skewed",
}

# Ambiguous tasks (Amb == 1): the participant saw B's outcomes but not their
# probabilities.
OPTION_AMBIGUOUS_TEMPLATE = (
    "Option {label} paid {outcomes}, but the probabilities of these "
    "outcomes were not shown to the participant (the participant only knew "
    "they summed to 100% and stayed fixed for the whole task)."
)
HIDDEN_TRUTH_TEMPLATE = (
    "Hidden from the participant, but generating the actual draws and "
    "feedback: Option {label} really paid {distribution}."
)

# How the two options' random draws were related (Corr). Participants were
# never told this; they could only pick it up from feedback.
CORRELATION_ENGLISH = {
    -1: (
        "On each trial the two options' random draws were negatively related: "
        "when one option drew a relatively high outcome, the other tended to "
        "draw a relatively low one."
    ),
    0: "On each trial the two options' outcomes were drawn independently.",
    1: (
        "On each trial the two options' random draws were positively related: "
        "both options tended to draw relatively high, or relatively low, "
        "outcomes together."
    ),
}
CORRELATION_NOT_TOLD = (
    "The participant was not told this and could only notice it from feedback."
)

# Choice history. The legend appears once; each earlier trial then gets one
# compact line such as "6: B → -17; A → -8." or "2: A; no feedback."
HISTORY_HEADER = (
    "Choice history (trial number: option chosen → points received; "
    "other option → points it would have paid):"
)
HISTORY_FEEDBACK_LINE = "{trial}{rt}: {choice} → {received}; {other} → {forgone}."
HISTORY_NO_FEEDBACK_LINE = "{trial}{rt}: {choice}; no feedback."
NO_HISTORY_TEXT = "No previous choices were provided."

# The final question. Kept last, with the answer format stated explicitly.
QUESTION_TEMPLATE = (
    "Question: which option did the participant choose on trial {trial} of 25 "
    "(block {block})? Answer with exactly one letter: A or B."
)


# =============================================================================
# Value parsing helpers
# =============================================================================

def _as_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    if hasattr(row, "to_dict"):
        return dict(row.to_dict())
    raise TypeError("row must be a mapping or provide a to_dict() method")


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def _fraction(value: Any, field: str) -> Fraction:
    """Parse a number exactly; fractions avoid float artefacts like 0.30000004."""
    if _missing(value):
        raise ValueError(f"Missing value in {field}")
    try:
        return Fraction(str(value).strip())
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid numeric value for {field}: {value!r}") from exc


def _integer(value: Any, field: str) -> int:
    number = _fraction(value, field)
    if number.denominator != 1:
        raise ValueError(f"Expected an integer in {field}, got {value!r}")
    return number.numerator


def _clean_text(value: Any) -> str:
    return "" if _missing(value) else str(value).strip()


def _normalized_shape(shape: Any) -> str:
    """Map any spelling of a LotShape value to 'symm', 'r-skew', 'l-skew' or ''."""
    text = _clean_text(shape).lower().replace("_", "-")
    if text in {"", "-", "none"}:
        return ""
    aliases = {
        "symm": "symm", "symmetric": "symm",
        "r-skew": "r-skew", "rskew": "r-skew",
        "right-skew": "r-skew", "right-skewed": "r-skew",
        "l-skew": "l-skew", "lskew": "l-skew",
        "left-skew": "l-skew", "left-skewed": "l-skew",
    }
    if text not in aliases:
        raise ValueError(f"Unknown CPC18 lottery shape: {shape!r}")
    return aliases[text]


def _format_number(value: Fraction | int | float) -> str:
    number = value if isinstance(value, Fraction) else Fraction(str(value))
    if number.denominator == 1:
        return f"{number.numerator:,}"
    return f"{float(number):,.6f}".rstrip("0").rstrip(".")


def _format_probability(probability: Fraction) -> str:
    percentage = float(probability * 100)
    return f"{percentage:.6f}".rstrip("0").rstrip(".") + "%"


def _format_points(value: Fraction) -> str:
    """Format a payoff with its unit, e.g. '-8 points' or '1 point'."""
    unit = "point" if abs(value) == 1 else "points"
    return f"{_format_number(value)} {unit}"


def _join_or(items: Iterable[str]) -> str:
    values = list(items)
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} or {values[1]}"
    return ", ".join(values[:-1]) + f", or {values[-1]}"


def _ordinal(number: int) -> str:
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


# =============================================================================
# Reconstructing outcome distributions
#
# A CPC18 option pays its "high" branch with probability pH and its low
# outcome L otherwise. The high branch is either a single fixed payoff H, or
# a LotNum-outcome lottery with expected value H whose shape is LotShape
# (Plonsky et al., 2025, supplementary information):
#   - 'Symm':   LotNum outcomes centred on H with binomial probabilities;
#   - 'R-Skew': truncated geometric probabilities 1/2, 1/4, ... with outcomes
#               spread increasingly far above H (long right tail);
#   - 'L-Skew': the mirror image, with a long left tail below H.
# =============================================================================

def _inner_lottery(
    high_mean: Fraction, shape: str, number_of_outcomes: int
) -> list[tuple[Fraction, Fraction]]:
    """Return the high branch as ``(outcome, conditional probability)`` pairs."""
    if number_of_outcomes < 1:
        raise ValueError("LotNum must be at least 1")

    kind = _normalized_shape(shape)
    if number_of_outcomes == 1 or not kind:
        return [(high_mean, Fraction(1))]

    if kind == "symm":
        k = number_of_outcomes - 1
        return [
            (
                high_mean + Fraction(j, 1) - Fraction(k, 2),
                Fraction(math.comb(k, j), 2**k),
            )
            for j in range(k + 1)
        ]

    direction = 1 if kind == "r-skew" else -1
    # Truncated geometric probabilities 1/2, 1/4, ..., with the final
    # probability doubled so they sum to one; offsets keep the mean at H.
    result: list[tuple[Fraction, Fraction]] = []
    for index in range(1, number_of_outcomes + 1):
        offset = direction * Fraction(2**index - (number_of_outcomes + 1), 1)
        probability = (
            Fraction(1, 2**index)
            if index < number_of_outcomes
            else Fraction(1, 2 ** (number_of_outcomes - 1))
        )
        result.append((high_mean + offset, probability))
    return result


def option_distribution(
    high_mean: Any,
    high_probability: Any,
    low_outcome: Any,
    lottery_shape: Any,
    number_of_lottery_outcomes: Any,
) -> list[tuple[Fraction, Fraction]]:
    """Return an option's final outcome distribution as ``(outcome, probability)``.

    Pairs are sorted by ascending outcome; zero-probability outcomes are
    dropped.
    """
    high = _fraction(high_mean, "H")
    p_high = _fraction(high_probability, "pH")
    low = _fraction(low_outcome, "L")
    count = _integer(number_of_lottery_outcomes, "LotNum")
    if not Fraction(0) <= p_high <= Fraction(1):
        raise ValueError(f"pH must lie between 0 and 1, got {high_probability!r}")

    combined: defaultdict[Fraction, Fraction] = defaultdict(Fraction)
    for outcome, conditional_probability in _inner_lottery(high, lottery_shape, count):
        combined[outcome] += p_high * conditional_probability
    combined[low] += 1 - p_high

    return sorted(
        (pair for pair in combined.items() if pair[1]),
        key=lambda pair: pair[0],
    )


# =============================================================================
# English fragments for each prompt section
# =============================================================================

def _participant_text(values: Mapping[str, Any], include_subject_id: bool) -> str:
    age = _integer(values["Age"], "Age")
    gender = GENDER_ENGLISH.get(_clean_text(values["Gender"]).lower())
    gender_age = f"{gender}, age {age}" if gender else f"age {age}"
    text = PARTICIPANT_TEMPLATE.format(
        gender_age=gender_age, location=_clean_text(values["Location"])
    )
    if include_subject_id:
        text = text[:-1] + f" (ID {_clean_text(values['SubjID'])})."
    return text


def _option_text(
    label: str,
    high: Any,
    p_high: Any,
    low: Any,
    shape: Any,
    count: Any,
    *,
    probabilities_hidden: bool,
) -> str:
    """Describe one option, highest outcome first, as on the CPC18 screen.

    With ``probabilities_hidden`` (Option B in ambiguous tasks), the first
    sentence lists only the outcomes the participant saw; the true
    distribution follows, explicitly labelled as hidden.
    """
    distribution = option_distribution(high, p_high, low, shape, count)
    screen_order = list(reversed(distribution))  # highest outcome first

    full_text = _join_or(
        OUTCOME_TEMPLATE.format(
            outcome=_format_points(outcome),
            probability=_format_probability(probability),
        )
        for outcome, probability in screen_order
    )

    if probabilities_hidden:
        sentences = [
            OPTION_AMBIGUOUS_TEMPLATE.format(
                label=label,
                outcomes=_join_or(
                    _format_points(outcome) for outcome, _ in screen_order
                ),
            ),
            HIDDEN_TRUTH_TEMPLATE.format(label=label, distribution=full_text),
        ]
    elif len(distribution) == 1:
        sentences = [
            OPTION_SURE_TEMPLATE.format(
                label=label, outcome=_format_points(distribution[0][0])
            )
        ]
    else:
        sentences = [OPTION_RISKY_TEMPLATE.format(label=label, distribution=full_text)]

    lottery_size = _integer(count, "LotNum")
    lottery_shape = _normalized_shape(shape)
    if lottery_size > 1 and lottery_shape:
        sentences.append(
            LOTTERY_SHAPE_TEMPLATE.format(
                label=label,
                count=lottery_size,
                shape=LOT_SHAPE_ENGLISH[lottery_shape],
                mean=_format_number(_fraction(high, "H")),
            )
        )
    return " ".join(sentences)


def _correlation_text(correlation: int) -> str:
    if correlation not in CORRELATION_ENGLISH:
        raise ValueError(f"Corr must be -1, 0, or 1, got {correlation!r}")
    text = CORRELATION_ENGLISH[correlation]
    if correlation != 0:
        text += " " + CORRELATION_NOT_TOLD
    return text


def _task_structure_text(values: Mapping[str, Any]) -> str:
    """Describe both options, ambiguity, and the options' correlation."""
    ambiguous = bool(_integer(values["Amb"], "Amb"))
    lines = [
        _option_text(
            "A", values["Ha"], values["pHa"], values["La"],
            values["LotShapeA"], values["LotNumA"],
            probabilities_hidden=False,
        ),
        _option_text(
            "B", values["Hb"], values["pHb"], values["Lb"],
            values["LotShapeB"], values["LotNumB"],
            probabilities_hidden=ambiguous,
        ),
        _correlation_text(_integer(values["Corr"], "Corr")),
    ]
    return "\n".join(lines)


def _history_records(rows: Any) -> list[dict[str, Any]]:
    if rows is None:
        return []
    if hasattr(rows, "to_dict") and not isinstance(rows, Mapping):
        try:
            records = rows.to_dict(orient="records")
        except TypeError:
            records = None
        if records is not None:
            return [dict(record) for record in records]
    return [_as_dict(row) for row in rows]


def _history_text(rows: list[dict[str, Any]], include_response_time: bool) -> str:
    """Write exactly one compact line per prior trial, in the supplied order."""
    lines: list[str] = []
    for row in rows:
        trial = _integer(row["Trial"], "Trial")
        chose_b = _integer(row["B"], "B")
        choice, other = ("B", "A") if chose_b else ("A", "B")
        rt = ""
        if include_response_time:
            rt = f" [{_format_number(_fraction(row['RT'], 'RT'))} ms]"
        if _integer(row["Feedback"], "Feedback"):
            lines.append(
                HISTORY_FEEDBACK_LINE.format(
                    trial=trial,
                    rt=rt,
                    choice=choice,
                    received=_format_number(_fraction(row["Payoff"], "Payoff")),
                    other=other,
                    forgone=_format_number(_fraction(row["Forgone"], "Forgone")),
                )
            )
        else:
            lines.append(
                HISTORY_NO_FEEDBACK_LINE.format(trial=trial, rt=rt, choice=choice)
            )
    return "\n".join(lines)


# =============================================================================
# Selecting rows and history from a DataFrame
# =============================================================================

def get_cpc18_row_and_history(
    dataframe: Any,
    row_number: int,
) -> tuple[Any, Any]:
    """Return ``(history, target_row)`` from a CPC18 pandas DataFrame.

    ``row_number`` is one-based: ``1`` selects the first data row, matching the
    ordinary meaning of "row 1" after loading the CSV header with pandas.

    The history is every earlier trial of the same participant and task. It
    never contains the target or a later trial.
    """
    if not hasattr(dataframe, "iloc") or not hasattr(dataframe, "columns"):
        raise TypeError("dataframe must be a pandas DataFrame")
    missing_columns = sorted(REQUIRED_COLUMNS - set(dataframe.columns))
    if missing_columns:
        raise KeyError("Missing CPC18 columns: " + ", ".join(missing_columns))
    if isinstance(row_number, bool) or not isinstance(row_number, int):
        raise TypeError("row_number must be an integer")
    if not 1 <= row_number <= len(dataframe):
        raise IndexError(f"row_number must be between 1 and {len(dataframe)}")

    row = dataframe.iloc[row_number - 1].copy()
    same_participant = dataframe["SubjID"] == row["SubjID"]
    same_set = dataframe["Set"] == row["Set"]
    same_task = dataframe["GameID"] == row["GameID"]
    earlier_trial = dataframe["Trial"] < row["Trial"]
    mask = same_participant & same_set & same_task & earlier_trial
    history = dataframe.loc[mask].sort_values("Trial", kind="stable").copy()
    return history, row


# =============================================================================
# Public prompt builders
# =============================================================================

def cpc18_row_to_english(
    row: Any,
    *,
    history_rows: Any = None,
    include_general_introduction: bool = True,
    include_participant_details: bool = True,
    include_task_structure: bool = True,
    include_subject_id: bool = False,
    include_history_response_times: bool = False,
    strict: bool = True,
) -> str:
    """Build a prediction prompt for one CPC18 target trial.

    Parameters
    ----------
    row:
        The target row. Its choice (``B``/``Button``), response time, and trial
        outcomes are never included because the choice is the prediction label.
    history_rows:
        History to render, in the supplied order. This may be a list of
        mappings or a pandas DataFrame. Use ``get_cpc18_row_and_history`` to
        select it. With ``strict=True``, mismatched, current, or future rows
        raise an error.
    include_general_introduction:
        Include a short explanation of the CPC18 repeated-choice procedure.
    include_participant_details:
        Include age, gender, and experimental location.
    include_task_structure:
        Include the task's position in the session, both options' outcome
        distributions, ambiguity, and the correlation between the options.
    include_subject_id:
        Append the participant's ID to the participant sentence.
    include_history_response_times:
        Show each history trial's response time in milliseconds.
    strict:
        Reject mismatched history rows and inconsistent trial/block schedules.
    """
    values = _as_dict(row)
    missing_columns = sorted(REQUIRED_COLUMNS - values.keys())
    if missing_columns:
        raise KeyError("Missing CPC18 columns: " + ", ".join(missing_columns))

    game = _clean_text(values["GameID"])
    trial = _integer(values["Trial"], "Trial")
    block = _integer(values["block"], "block")
    feedback = _integer(values["Feedback"], "Feedback")

    if strict:
        expected_block = (trial - 1) // 5 + 1
        expected_feedback = int(trial >= 6)
        errors = []
        if block != expected_block:
            errors.append(f"block={block}, expected {expected_block} from Trial={trial}")
        if feedback != expected_feedback:
            errors.append(
                f"Feedback={feedback}, expected {expected_feedback} from Trial={trial}"
            )
        if errors:
            raise ValueError("Inconsistent CPC18 row: " + "; ".join(errors))

    paragraphs: list[str] = []
    if include_general_introduction:
        paragraphs.append(GENERAL_INTRODUCTION)

    if include_participant_details:
        paragraphs.append(_participant_text(values, include_subject_id))

    if include_task_structure:
        condition = _clean_text(values["Condition"])
        condition_text = (
            CONDITION_BY_PROB
            if condition.lower() == "byprob"
            else CONDITION_OTHER_TEMPLATE.format(condition=condition)
        )
        order_text = TASK_CONTEXT_TEMPLATE.format(
            order=_ordinal(_integer(values["Order"], "Order"))
        )
        paragraphs.append(
            f"{order_text} {condition_text}\n{_task_structure_text(values)}"
        )

    supplied_history = _history_records(history_rows)
    if strict:
        for previous in supplied_history:
            previous_trial = _integer(previous["Trial"], "Trial")
            same_subject = (
                _clean_text(previous["SubjID"]) == _clean_text(values["SubjID"])
            )
            same_set = _clean_text(previous["Set"]) == _clean_text(values["Set"])
            same_game = _clean_text(previous["GameID"]) == game
            if not same_subject or not same_set or not same_game:
                raise ValueError(
                    "history_rows must match the target SubjID, Set, and GameID"
                )
            if previous_trial >= trial:
                raise ValueError(
                    "history_rows must contain only trials before the target trial"
                )

    if supplied_history:
        paragraphs.append(
            HISTORY_HEADER
            + "\n"
            + _history_text(supplied_history, include_history_response_times)
        )
    else:
        paragraphs.append(NO_HISTORY_TEXT)

    paragraphs.append(QUESTION_TEMPLATE.format(trial=trial, block=block))

    return "\n".join(paragraphs)


def iter_cpc18_csv_as_english(
    csv_path: str | Path,
    **translation_options: Any,
) -> Iterator[str]:
    """Stream leakage-free prediction prompts for every row of a CPC18 CSV."""
    histories: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (
                _clean_text(row["SubjID"]),
                _clean_text(row["Set"]),
                _clean_text(row["GameID"]),
            )
            yield cpc18_row_to_english(
                row,
                history_rows=histories[key],
                **translation_options,
            )
            histories[key].append(row)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert CPC18 rows to leakage-free prediction prompts."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--include-subject-id", action="store_true")
    parser.add_argument("--no-general-introduction", action="store_true")
    parser.add_argument("--no-participant-details", action="store_true")
    parser.add_argument("--no-task-structure", action="store_true")
    parser.add_argument("--include-history-response-times", action="store_true")
    args = parser.parse_args()

    with args.output_path.open("w", encoding="utf-8") as output:
        for text in iter_cpc18_csv_as_english(
            args.csv_path,
            include_subject_id=args.include_subject_id,
            include_general_introduction=not args.no_general_introduction,
            include_participant_details=not args.no_participant_details,
            include_task_structure=not args.no_task_structure,
            include_history_response_times=args.include_history_response_times,
        ):
            output.write(text.replace("\n", " ") + "\n")
