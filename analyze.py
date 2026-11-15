"""Score every collected LLM condition and compare it with cognitive models.

Reads outputs/<condition>/predictions.csv (written by openai_batch_collect.py)
and the cognitive-model forecasts in the data folder, then writes tables, a
figure and REPORT.md to outputs/analysis/.

All predictors are scored on the same trials: those with a valid prediction in
every collected condition. Uncertainty comes from resampling participants,
because trials of one participant are not independent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cognitive_models import forecast_probabilities, load_forecasts, reference_baselines
from experiment_config import DATA_DIR, OUTPUT_DIR

TRIAL_KEYS = ["SubjID", "Set", "GameID", "Trial"]
CELL_KEYS = ["GameID", "block"]
FACTORS = ["model", "effort", "instruction"]
NLL_CLIP = 1e-6


def ci(values, repeats: int = 3000, seed: int = 20261011) -> list[float]:
    """Return a 95% bootstrap interval for the mean of ``values``."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def score_rows(observed_b, probability_b) -> pd.DataFrame:
    """Return per-trial Brier score, accuracy and clipped negative log likelihood."""
    y = np.asarray(observed_b, dtype=float)
    p = np.asarray(probability_b, dtype=float)
    observed_probability = np.where(y == 1, p, 1 - p)
    return pd.DataFrame(
        {
            "Brier": (p - y) ** 2,
            # A probability of exactly 0.5 earns the expected accuracy of a coin flip.
            "accuracy": np.where(p == 0.5, 0.5, ((p > 0.5) == (y == 1)).astype(float)),
            "NLL": -np.log(np.clip(observed_probability, NLL_CLIP, 1)),
        }
    )


def llm_probability(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return p(B) and whether it came from token log-probabilities.

    Without log-probabilities the hard A/B answer is used as a 0/1 probability.
    """
    from_logprob = frame.log_likelihood_B.notna().to_numpy()
    probability = np.where(
        from_logprob,
        np.exp(frame.log_likelihood_B.astype(float)),
        frame.model_choice.astype(float),
    )
    return probability, from_logprob


def load_conditions(output_dir: Path) -> dict[str, pd.DataFrame]:
    """Return each collected condition's rows, indexed by trial."""
    loaded = {}
    for path in sorted(Path(output_dir).glob("*/predictions.csv")):
        frame = pd.read_csv(path)
        if frame.duplicated(TRIAL_KEYS).any():
            raise ValueError(f"Duplicate trials in {path}")
        # Index levels get their own names so they never clash with the columns.
        frame.index = pd.MultiIndex.from_frame(
            frame[TRIAL_KEYS], names=[f"index_{key}" for key in TRIAL_KEYS]
        )
        loaded[path.parent.name] = frame
    return loaded


def build_predictors(conditions: dict[str, pd.DataFrame], data_dir: Path):
    """Return the common trials and every predictor's p(B) on them."""
    valid = [
        frame.index[frame.batch_error.isna() & frame.model_choice.notna()]
        for frame in conditions.values()
    ]
    common = valid[0]
    for index in valid[1:]:
        common = common.intersection(index)
    if common.empty:
        raise ValueError("No trial has a valid prediction in every condition")
    common = common.sort_values()

    first = next(iter(conditions.values()))
    data_columns = TRIAL_KEYS + ["block", "B"]
    trials = first.loc[common, data_columns].reset_index(drop=True)

    predictors = []
    for name, frame in conditions.items():
        rows = frame.loc[common]
        probability, from_logprob = llm_probability(rows)
        predictors.append(
            {
                "predictor": name,
                "kind": "LLM",
                **{factor: str(rows[factor].iloc[0]) for factor in FACTORS},
                "probability_B": probability,
                # Score the returned answer itself, not a thresholded probability.
                "accuracy": (rows.model_choice.to_numpy() == rows.B.to_numpy()).astype(float),
                "logprob_share": float(from_logprob.mean()),
            }
        )

    raw_path = Path(data_dir) / "rawData_Comp_All.csv"
    raw = pd.read_csv(raw_path) if raw_path.is_file() else None
    games = (raw if raw is not None else first).GameID.unique()
    forecasts = load_forecasts(data_dir, games)
    if forecasts.empty:
        print(f"Warning: no cognitive-model forecast files found in {data_dir}")
    for name, values in forecast_probabilities(trials, forecasts).items():
        predictors.append({"predictor": name, "kind": "cognitive model",
                           "probability_B": values.to_numpy()})

    # Baselines need complete sequences, so compute them before restricting rows.
    baselines = reference_baselines(first[data_columns]).loc[common]
    for name, values in baselines.items():
        predictors.append({"predictor": name, "kind": "reference baseline",
                           "probability_B": values.to_numpy()})
    return trials, predictors, raw


def analyze(output_dir: Path = OUTPUT_DIR, data_dir: Path = DATA_DIR) -> pd.DataFrame | None:
    conditions = load_conditions(output_dir)
    if not conditions:
        print(f"No predictions.csv files found in {output_dir}")
        return None
    trials, predictors, raw = build_predictors(conditions, data_dir)
    analysis_dir = Path(output_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    subject = trials.SubjID.to_numpy()

    # ---- Individual-choice metrics ------------------------------------------
    metric_rows = []
    subject_brier = {}
    trial_brier = {}
    for predictor in predictors:
        scores = score_rows(trials.B, predictor["probability_B"])
        if "accuracy" in predictor:
            scores["accuracy"] = predictor["accuracy"]
        name = predictor["predictor"]
        subject_brier[name] = scores.Brier.groupby(subject).mean()
        trial_brier[name] = scores.Brier.groupby(trials.Trial.to_numpy()).mean()
        low, high = ci(subject_brier[name])
        metric_rows.append(
            {
                "predictor": name,
                "kind": predictor["kind"],
                **{factor: predictor.get(factor) for factor in FACTORS},
                "n": len(trials),
                "participants": int(trials.SubjID.nunique()),
                "Brier": float(scores.Brier.mean()),
                "Brier_CI_low": low,
                "Brier_CI_high": high,
                "accuracy": float(scores.accuracy.mean()),
                "NLL": float(scores.NLL.mean()),
                "predicted_B_rate": float(np.mean(predictor["probability_B"])),
                "observed_B_rate": float(trials.B.mean()),
                "logprob_share": predictor.get("logprob_share"),
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values("Brier").reset_index(drop=True)
    metrics.to_csv(analysis_dir / "metrics.csv", index=False)
    pd.DataFrame(trial_brier).rename_axis("Trial").to_csv(analysis_dir / "by_trial.csv")

    # ---- Each LLM condition against each cognitive model --------------------
    llm_names = [p["predictor"] for p in predictors if p["kind"] == "LLM"]
    cognitive_names = [p["predictor"] for p in predictors if p["kind"] == "cognitive model"]
    paired_rows = []
    for llm in llm_names:
        for cognitive in cognitive_names:
            difference = subject_brier[llm] - subject_brier[cognitive]
            low, high = ci(difference)
            paired_rows.append(
                {"LLM": llm, "cognitive_model": cognitive,
                 "Brier_difference": float(difference.mean()),
                 "CI_low": low, "CI_high": high, "participants": len(difference)}
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(analysis_dir / "paired_vs_cognitive.csv", index=False)

    # ---- Effect of model, effort and instruction ----------------------------
    llm_metrics = metrics[metrics.kind.eq("LLM")]
    by_factor = pd.concat(
        [
            llm_metrics.groupby(factor)
            .agg(conditions=("Brier", "size"), Brier=("Brier", "mean"),
                 accuracy=("accuracy", "mean"), NLL=("NLL", "mean"))
            .rename_axis("level").reset_index().assign(factor=factor)
            for factor in FACTORS
        ],
        ignore_index=True,
    )[["factor", "level", "conditions", "Brier", "accuracy", "NLL"]]
    by_factor.to_csv(analysis_dir / "by_factor.csv", index=False)

    # ---- Aggregate choice rates (the CPC18 competition metric) --------------
    target_source = "all participants in rawData_Comp_All.csv" if raw is not None else "analyzed trials only"
    target = (raw if raw is not None else trials).groupby(CELL_KEYS).B.mean().rename("observed")
    aggregate_rows = []
    for predictor in predictors:
        predicted = (
            pd.Series(predictor["probability_B"]).groupby(
                [trials.GameID.to_numpy(), trials.block.to_numpy()]).mean()
        )
        predicted.index.names = CELL_KEYS
        cells = pd.concat([predicted.rename("predicted"), target], axis=1, join="inner")
        aggregate_rows.append(
            {"predictor": predictor["predictor"], "kind": predictor["kind"],
             "cells": len(cells), "MSE": float(((cells.predicted - cells.observed) ** 2).mean())}
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values("MSE").reset_index(drop=True)
    aggregate.to_csv(analysis_dir / "aggregate_mse.csv", index=False)

    plot_results(metrics, trial_brier, analysis_dir / "results.png")
    write_report(analysis_dir / "REPORT.md", metrics, paired, by_factor, aggregate, target_source)
    print(metrics[["predictor", "kind", "n", "Brier", "accuracy", "NLL"]].to_string(index=False))
    print(f"Wrote analysis to {analysis_dir}")
    return metrics


def plot_results(metrics: pd.DataFrame, trial_brier: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"LLM": "#c65a27", "cognitive model": "#217a83", "reference baseline": "#888888"}
    ordered = metrics.sort_values("Brier", ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.32 * len(ordered) + 1.5)))
    for kind, group in ordered.groupby("kind", sort=False):
        # A bootstrap bound can fall on the wrong side of the mean by rounding.
        below = np.clip((group.Brier - group.Brier_CI_low).to_numpy(), 0, None)
        above = np.clip((group.Brier_CI_high - group.Brier).to_numpy(), 0, None)
        axes[0].errorbar(group.Brier.to_numpy(), group.index.to_numpy(), xerr=[below, above],
                         fmt="o", color=colors[kind], label=kind, capsize=2)
    axes[0].set_yticks(range(len(ordered)), ordered.predictor)
    axes[0].set(xlabel="Individual Brier score (lower is better; 95% participant CI)",
                title="All predictors")
    axes[0].grid(axis="x", alpha=0.2)
    axes[0].legend(frameon=False)

    best_llm = metrics[metrics.kind.eq("LLM")].predictor.iloc[0]
    shown = [best_llm] + metrics[~metrics.kind.eq("LLM")].predictor.tolist()
    for name in shown:
        curve = trial_brier[name]
        kind = metrics.set_index("predictor").kind[name]
        axes[1].plot(curve.index, curve.values, label=name, color=None if kind != "LLM" else colors["LLM"],
                     linestyle="-" if kind != "reference baseline" else "--")
    axes[1].set(xlabel="Trial", ylabel="Individual Brier score",
                title="By trial: best LLM condition and comparison models")
    axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="upper center",
                   bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(frame.columns) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |"
              for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def write_report(path, metrics, paired, by_factor, aggregate, target_source) -> None:
    n, participants = int(metrics.n.iloc[0]), int(metrics.participants.iloc[0])
    conditions = int(metrics.kind.eq("LLM").sum())
    lines = [
        "# CPC18: LLM conditions vs. cognitive models", "",
        f"{conditions} LLM condition(s), scored on the same {n:,} trials from "
        f"{participants} participants. Every LLM prompt contained all of the "
        "participant's earlier trials in the task.", "",
        "## Individual choices", "",
        "Brier is the squared error between p(B) and the observed choice (0 or 1). "
        f"NLL clips the observed choice's probability at {NLL_CLIP:g}. `logprob_share` "
        "is the share of an LLM's trials whose p(B) came from token log-probabilities; "
        "the rest use the hard A/B answer as 0 or 1.", "",
        markdown_table(metrics.drop(columns=["n", "participants", "observed_B_rate"])), "",
        "## LLM minus cognitive model", "",
        "Difference in Brier score, paired by participant. Negative values favor the LLM.", "",
        markdown_table(paired) if len(paired) else "No cognitive-model forecasts were available.", "",
        "## Effect of model, effort and instruction", "",
        "Mean over the LLM conditions that share each level.", "",
        markdown_table(by_factor), "",
        "## Aggregate choice rates", "",
        "Mean squared error between each predictor's mean p(B) per problem and block "
        f"and the observed B rate ({target_source}). This is the CPC18 competition "
        "metric. The cognitive models forecast these rates from the task alone, while "
        "the LLM values average predictions that used individual histories.", "",
        markdown_table(aggregate), "",
        "![Results](results.png)", "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    analyze(args.outputs, args.data)
