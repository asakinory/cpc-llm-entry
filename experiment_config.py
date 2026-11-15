"""Experiment grid and shared settings.

Edit MODELS, EFFORTS and INSTRUCTIONS to change the grid. Every combination of
the three is one condition, and every condition sends one request per row of
INPUT_CSV, so the number of requests is

    len(MODELS) * len(EFFORTS) * len(INSTRUCTIONS) * rows in INPUT_CSV.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

# Both folders can be moved with environment variables, e.g. to keep the data
# on another drive.
DATA_DIR = Path(os.environ.get("CPC18_DATA_DIR", PROJECT_DIR / "data"))
OUTPUT_DIR = Path(os.environ.get("CPC18_OUTPUT_DIR", PROJECT_DIR / "outputs"))

# The complete CPC18 individual-level test data (180,000 rows).
RAW_DATA_PATH = DATA_DIR / "rawData_Comp_All.csv"
# The rows actually sent to the models. Create it with balanced_subset.py, or
# point this at RAW_DATA_PATH to run every row.
INPUT_CSV = DATA_DIR / "balanced_6000.csv"


# =============================================================================
# THE GRID -- edit these three settings
# =============================================================================

# Model size: list the models to compare, e.g. from smallest to largest.
MODELS = ["gpt-5.6-luna"]

# Reasoning effort levels passed as `reasoning_effort`.
EFFORTS = ["none", "low", "medium", "high"]

# Instruction variants, sent as the developer message. Every variant ends with
# the same answer format so the A/B output token can be scored.
ANSWER_FORMAT = (
    "Output exactly one character: A or B. "
    "Do not output punctuation, formatting, or explanation."
)
INSTRUCTIONS = {
    "minimal": "Choose one option. " + ANSWER_FORMAT,
    "predict_human": (
        "You are predicting the behavior of a human participant in a "
        "decision-making experiment. Answer with the option this participant "
        "most likely chose on the target trial, not the option you would "
        "choose yourself. " + ANSWER_FORMAT
    ),
    "behavioral_hint": (
        "You are predicting the behavior of a human participant in a "
        "decision-making experiment. Answer with the option this participant "
        "most likely chose on the target trial, not the option you would "
        "choose yourself. People often deviate from maximizing expected "
        "value: they are sensitive to recent feedback and tend to underweight "
        "rare outcomes when they learn from experience. " + ANSWER_FORMAT
    ),
}


# =============================================================================
# Pricing, in USD per million tokens (standard tier). Costs are reported as
# missing for models without an entry.
# =============================================================================
PRICING = {
    "gpt-5.6-luna": {
        "input": 0.20,
        "cached_input": 0.02,
        "cache_write": 0.25,
        "output": 1.20,
        "long_context_threshold": 272_000,
    },
}
BATCH_PRICE_MULTIPLIER = 0.5


@dataclass(frozen=True)
class Condition:
    """One cell of the grid: a model, an effort level and an instruction."""

    model: str
    effort: str
    instruction: str  # a key of INSTRUCTIONS

    @property
    def instruction_text(self) -> str:
        return INSTRUCTIONS[self.instruction]

    @property
    def id(self) -> str:
        """Folder-safe name, e.g. 'gpt-5.6-luna__medium__predict_human'."""
        parts = (self.model, self.effort, self.instruction)
        return "__".join(re.sub(r"[^A-Za-z0-9._-]", "-", part) for part in parts)


def conditions(
    models: list[str] | None = None,
    efforts: list[str] | None = None,
    instructions: list[str] | None = None,
) -> list[Condition]:
    """Return the grid, optionally restricted to the given values."""
    instruction_names = instructions or list(INSTRUCTIONS)
    unknown = sorted(set(instruction_names) - set(INSTRUCTIONS))
    if unknown:
        raise ValueError("Unknown instruction variant(s): " + ", ".join(unknown))
    return [
        Condition(model, effort, instruction)
        for model, effort, instruction in product(
            models or MODELS, efforts or EFFORTS, instruction_names
        )
    ]
