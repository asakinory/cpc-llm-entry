"""Check every submitted condition and write its predictions when ready.

For each outputs/<condition>/manifest.json whose batches have finished, this
writes outputs/<condition>/predictions.csv: the input rows plus the model's
answer and A/B log likelihoods. Run it again later for unfinished batches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from experiment_config import DATA_DIR, OUTPUT_DIR
from openai_batch_submit import MANIFEST_VERSION, file_sha256
from predict_request_openai import choice_log_likelihoods, request_cost


# Usually leave these as None; routing is read from the manifest/.env.
OPENAI_PROJECT = None
OPENAI_ORGANIZATION = None

TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}
RESULT_COLUMNS = (
    "condition",
    "model",
    "effort",
    "instruction",
    "prompt",
    "model_choice",
    "model_answer",
    "log_likelihood_A",
    "log_likelihood_B",
    "estimated_cost_usd",
    "model_name",
    "batch_error",
)


def _file_text(client, file_id: str) -> str:
    content = client.files.content(file_id)
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text
    raw = content.read()
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def _jsonl_records(text: str) -> Iterator[dict[str, Any]]:
    for line in text.splitlines():
        if line.strip():
            yield json.loads(line)


def _row_number(custom_id: str) -> int:
    prefix = "row-"
    if not custom_id.startswith(prefix):
        raise ValueError(f"Unexpected custom_id in batch result: {custom_id!r}")
    return int(custom_id[len(prefix) :])


def parse_response_body(
    body: dict[str, Any], pricing: dict[str, Any] | None
) -> dict[str, Any]:
    """Turn one Chat Completions response body into result columns."""
    text = body["choices"][0]["message"].get("content") or ""
    answer = text.strip().upper()
    if answer not in {"A", "B"}:
        raise ValueError(f"Expected exactly A or B, got {text!r}")
    log_likelihood_a, log_likelihood_b = choice_log_likelihoods(body)
    multiplier = (pricing or {}).get("batch_multiplier", 1.0)
    return {
        "model_choice": {"A": 0, "B": 1}[answer],
        "model_answer": answer,
        "log_likelihood_A": log_likelihood_a,
        "log_likelihood_B": log_likelihood_b,
        "estimated_cost_usd": request_cost(body.get("usage"), pricing, multiplier),
        "model_name": body.get("model"),
        "batch_error": None,
    }


def _user_prompt(request_body: dict[str, Any]) -> str:
    """Return the exact user-message text from a submitted request."""
    for message in request_body.get("messages") or []:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    raise ValueError("Batch request does not contain a user prompt")


def _collect_prompts(client, batches: list[Any]) -> dict[int, str]:
    """Read prompts from the uploaded inputs so they match submitted requests."""
    prompts: dict[int, str] = {}
    for batch in batches:
        for record in _jsonl_records(_file_text(client, batch.input_file_id)):
            prompts[_row_number(record["custom_id"])] = _user_prompt(record["body"])
    return prompts


def _collect_results(
    client, batches: list[Any], pricing: dict[str, Any] | None
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    for batch in batches:
        if batch.output_file_id:
            for record in _jsonl_records(_file_text(client, batch.output_file_id)):
                row_number = _row_number(record["custom_id"])
                response = record.get("response") or {}
                body = response.get("body") or {}
                error = record.get("error")
                if error or response.get("status_code") != 200:
                    results[row_number] = {
                        "batch_error": json.dumps(error or body.get("error") or response)
                    }
                    continue
                try:
                    results[row_number] = parse_response_body(body, pricing)
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    results[row_number] = {"batch_error": str(exc)}

        if batch.error_file_id:
            for record in _jsonl_records(_file_text(client, batch.error_file_id)):
                results[_row_number(record["custom_id"])] = {
                    "batch_error": json.dumps(record.get("error") or record)
                }
    return results


def _batch_level_error(batches: list[Any]) -> str | None:
    messages: list[str] = []
    for batch in batches:
        errors = getattr(batch, "errors", None)
        for error in getattr(errors, "data", []) or []:
            message = getattr(error, "message", None)
            if message:
                messages.append(f"{batch.id}: {message}")
        if batch.status != "completed" and not batch.output_file_id:
            fallback = f"{batch.id}: batch ended with status {batch.status}"
            if not any(message.startswith(f"{batch.id}:") for message in messages):
                messages.append(fallback)
    return " | ".join(messages) or None


def collect_condition(client, manifest_file: Path) -> pd.DataFrame | None:
    """Write predictions.csv for one condition; return None if not ready."""
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(f"Unsupported manifest version in {manifest_file}")

    input_path = Path(manifest["input_csv"])
    if not input_path.is_absolute():
        input_path = DATA_DIR / input_path
    if not input_path.is_file():
        raise FileNotFoundError(f"Original input CSV is missing: {input_path}")
    if file_sha256(input_path) != manifest["input_csv_sha256"]:
        raise ValueError(
            "The input CSV changed after submission; refusing to merge results"
        )

    batches = [client.batches.retrieve(batch_id) for batch_id in manifest["batch_ids"]]
    for batch in batches:
        counts = batch.request_counts
        count_text = ""
        if counts is not None:
            count_text = (
                f" ({counts.completed}/{counts.total} completed, {counts.failed} failed)"
            )
        print(f"  {batch.id}: {batch.status}{count_text}")
    if any(batch.status not in TERMINAL_BATCH_STATUSES for batch in batches):
        print("  Not ready yet. Run this collector again later.")
        return None

    data = pd.read_csv(input_path).reset_index(drop=True)
    if len(data) != manifest["row_count"]:
        raise ValueError("The input CSV row count no longer matches the manifest")

    results = _collect_results(client, batches, manifest.get("pricing"))
    prompts = _collect_prompts(client, batches)
    batch_error = _batch_level_error(batches)
    constant = {key: manifest[key] for key in ("condition", "model", "effort", "instruction")}
    empty = {column: None for column in RESULT_COLUMNS}
    result_rows = []
    for row_number in range(len(data)):
        result = {**empty, **constant, **results.get(row_number, {})}
        result["prompt"] = prompts.get(row_number)
        if row_number not in results:
            result["batch_error"] = batch_error or "No result returned by Batch API"
        result_rows.append(result)

    result_frame = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)
    result_frame["model_choice"] = pd.array(result_frame["model_choice"], dtype="Int64")
    augmented = pd.concat([data, result_frame], axis=1)
    output_path = manifest_file.with_name("predictions.csv")
    augmented.to_csv(output_path, index=False)

    cost = result_frame["estimated_cost_usd"].dropna().sum()
    failures = int(result_frame["batch_error"].notna().sum())
    print(f"  wrote {output_path} (estimated cost ${cost:,.4f}, {failures:,} failed rows)")
    return augmented


def collect() -> None:
    manifests = sorted(OUTPUT_DIR.glob("*/manifest.json"))
    if not manifests:
        print(f"No submitted conditions found in {OUTPUT_DIR}")
        return

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(Path(__file__).with_name(".env"), override=True)
    for manifest_file in manifests:
        print(manifest_file.parent.name)
        if manifest_file.with_name("predictions.csv").exists():
            print("  already collected")
            continue
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        client = OpenAI(
            project=OPENAI_PROJECT or manifest.get("project"),
            organization=OPENAI_ORGANIZATION or manifest.get("organization"),
        )
        collect_condition(client, manifest_file)


if __name__ == "__main__":
    collect()
