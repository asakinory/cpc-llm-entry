# LLM choice prediction on CPC18

This repository tests how well large language models predict individual human
choices in the CPC18 decisions-under-risk data, and compares them with
cognitive models.

For every trial, the model reads a plain-English description of the task and
**all of the participant's earlier choices and feedback in that task**, then
predicts whether the participant chose Option A or Option B on the next trial.
The experiment is a grid over three settings:

| Setting | What varies | Default |
| --- | --- | --- |
| Model size | Which model answers | `gpt-5.6-luna` |
| Reasoning effort | The `reasoning_effort` request parameter | `none`, `low`, `medium`, `high` |
| Instructions | The developer message | `minimal`, `predict_human`, `behavioral_hint` |

Each combination (a *condition*) is scored on the same trials and compared with:

- **Cognitive models:** BEAST, the CPC18 organizers' baseline, and BEAST-GB, the
  competition winner. Their published forecasts give a choice rate for each
  problem and block of five trials.
- **Reference baselines:** a constant 0.5, repeating the participant's previous
  choice, and the participant's smoothed rate of earlier B choices.

## Data

The `data/` folder is empty; the data is not redistributed here. Download
three files from the CPC18 project on OSF (<https://osf.io/vw2su/>, archive
<https://osf.io/download/2h9rx/>) and put them directly in `data/`:

| File in `data/` | Path inside the archive | Contents |
| --- | --- | --- |
| `rawData_Comp_All.csv` | `Final files for upload/CPC18/` | Individual choices in the competition's test set: 240 participants, 60 problems, 25 trials each (180,000 rows) |
| `BEAST_sd.csv` | `Final files for upload/CPC18/all Track 1 submissions/` | BEAST forecasts (60 problems x 5 blocks) |
| `BP47.csv` | `Final files for upload/CPC18/all Track 1 submissions/` | BEAST-GB forecasts (60 problems x 5 blocks) |

The two forecast files are optional. Without them the analysis still runs and
reports only the LLM conditions and the reference baselines.

To keep the data elsewhere, set the environment variable `CPC18_DATA_DIR`.
`CPC18_OUTPUT_DIR` does the same for the `outputs/` folder.

## Setup

Requires Python 3.10 or later.

```
pip install -r requirements.txt
```

Create a file named `.env` in the repository folder with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

## Running the experiment

1. **Choose the rows.** Create a balanced subset of the raw data. Each 6,000-row
   round holds one complete 25-trial task per participant and four participants
   per problem.

   ```
   python balanced_subset.py 6000
   ```

   This writes `data/balanced_6000.csv`, the default input. To use another
   file, change `INPUT_CSV` in `experiment_config.py` or pass `--input`.

2. **Check the grid.** Without `--submit`, the submit script only lists the
   conditions and the number of requests.

   ```
   python openai_batch_submit.py
   ```

   The default grid is 12 conditions, so 72,000 requests for a 6,000-row input.
   `--models`, `--efforts` and `--instructions` restrict the run, for example
   `--efforts low high --instructions minimal`.

3. **Submit.** This uploads the requests to the OpenAI Batch API and is billed
   to your account.

   ```
   python openai_batch_submit.py --submit
   ```

   Each condition gets a folder `outputs/<model>__<effort>__<instruction>/`
   with a `manifest.json`. Conditions that already have a manifest are skipped,
   so the command can be repeated after extending the grid.

4. **Collect.** Batches finish within 24 hours. Run the collector whenever you
   like; it writes `predictions.csv` for each finished condition.

   ```
   python openai_batch_collect.py
   ```

5. **Analyze.**

   ```
   python analyze.py
   ```

`python main.py` sends a single live request for one trial and prints the
prompt and the answer, which is a quick way to check the setup.

## Changing the grid

Edit the three lists at the top of [experiment_config.py](experiment_config.py):

- `MODELS`: add the model names to compare, for example from smallest to
  largest. Add the model's prices to `PRICING` to get cost estimates.
- `EFFORTS`: the reasoning-effort levels.
- `INSTRUCTIONS`: named developer messages. Every variant must end with
  `ANSWER_FORMAT`, because scoring relies on a one-character A/B answer.

The prompt text itself is built in
[cpc18_row_to_english.py](cpc18_row_to_english.py); all wording lives in the
"translation tables" at the top of that file.

## Results

`analyze.py` writes to `outputs/analysis/`:

| File | Contents |
| --- | --- |
| `REPORT.md` | All tables below in one readable report |
| `metrics.csv` | Per predictor: Brier score with a 95% interval, choice accuracy, negative log likelihood, predicted and observed B rate |
| `paired_vs_cognitive.csv` | Brier score of each LLM condition minus each cognitive model, paired by participant; negative favors the LLM |
| `by_factor.csv` | Mean scores by model, by effort and by instruction |
| `aggregate_mse.csv` | Error in predicting choice rates per problem and block, the CPC18 competition metric |
| `by_trial.csv` | Brier score by trial number |
| `results.png` | Brier score of every predictor, and scores by trial |

How the scores are computed:

- The LLM's probability of B comes from the log-probabilities of the answer
  token, normalized over A and B. If a response has no log-probabilities, the
  hard answer counts as probability 0 or 1; `logprob_share` in `metrics.csv`
  shows how often that happened.
- The Brier score is the squared difference between p(B) and the observed
  choice (0 or 1). Lower is better; a constant 0.5 scores 0.25.
- All predictors are scored on the same trials: those with a valid answer in
  every collected condition.
- Intervals are 95% bootstrap intervals that resample participants, because
  one participant's trials are not independent.

## Limits

- **The comparison is not like for like.** The BEAST and BEAST-GB forecasts
  depend only on the task and were made before the test data existed. The LLMs
  see each participant's own earlier choices. The reference baselines show how
  much that history alone is worth.
- **Possible contamination.** CPC18 data and results are public, so a model
  may have seen them during training.
- **Probabilities with reasoning.** Token log-probabilities describe the final
  answer token only, after any reasoning the model did.
- **Cost.** Requests grow with the grid, and prompts get longer with the trial
  number. Start with a small grid and a 6,000-row subset.

## Tests

The tests use made-up rows and make no API calls, so they run without the data:

```
python -m unittest
```

## References

- Plonsky, O., Apel, R., Ert, E., Tennenholtz, M., Bourgin, D., Peterson, J. C.,
  et al. (2025). Predicting human decisions with behavioural theories and
  machine learning. *Nature Human Behaviour*.
  <https://www.nature.com/articles/s41562-025-02267-6>
- Erev, I., Ert, E., Plonsky, O., Cohen, D., & Cohen, O. (2017). From anomalies
  to forecasts: Toward a descriptive model of decisions under risk, under
  ambiguity, and from experience. *Psychological Review, 124*(4), 369-409.
- Binz, M., Akata, E., Bethge, M., Brändle, F., Callaway, F., Coda-Forno, J.,
  et al. (2025). A foundation model to predict and capture human cognition.
  *Nature*.
