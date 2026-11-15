"""Submit the experiment grid to the OpenAI Batch API and exit without polling.

By default this is a dry run that only lists the conditions and request
counts. Add --submit to upload the requests, which is billed by OpenAI.

    python openai_batch_submit.py                      # dry run, whole grid
    python openai_batch_submit.py --efforts low high   # dry run, part of the grid
    python openai_batch_submit.py --submit             # submit the whole grid
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import pandas as pd

from experiment_config import (
    BATCH_PRICE_MULTIPLIER,
    DATA_DIR,
    INPUT_CSV,
    OUTPUT_DIR,
    PRICING,
    Condition,
    conditions,
)
from openai_batch_requests import build_prediction_requests


# Leave these as None when .env contains a project-scoped API key.
OPENAI_PROJECT = None
OPENAI_ORGANIZATION = None

MAX_REQUESTS_PER_BATCH = 50_000
MAX_BATCH_FILE_BYTES = 190_000_000
FILE_PROPAGATION_WAIT_SECONDS = 2.0
MANIFEST_VERSION = 2


def manifest_path(condition: Condition) -> Path:
    return OUTPUT_DIR / condition.id / "manifest.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stored_path(path: Path) -> str:
    """Store paths inside the data folder by name so manifests stay portable."""
    try:
        return path.resolve().relative_to(DATA_DIR.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_batch_input_files(
    data: pd.DataFrame, condition: Condition, directory: Path
) -> list[Path]:
    paths: list[Path] = []
    handle = None
    request_count = 0
    byte_count = 0

    try:
        for request in build_prediction_requests(data, condition):
            encoded = (
                json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            if len(encoded) > MAX_BATCH_FILE_BYTES:
                raise ValueError("One generated request exceeds the batch file limit")

            needs_new_file = handle is None or (
                request_count > 0
                and (
                    request_count >= MAX_REQUESTS_PER_BATCH
                    or byte_count + len(encoded) > MAX_BATCH_FILE_BYTES
                )
            )
            if needs_new_file:
                if handle is not None:
                    handle.close()
                path = directory / f"requests_{len(paths):04d}.jsonl"
                paths.append(path)
                handle = path.open("wb")
                request_count = 0
                byte_count = 0

            handle.write(encoded)
            request_count += 1
            byte_count += len(encoded)
    finally:
        if handle is not None:
            handle.close()

    if not paths:
        raise ValueError("The input CSV has no rows")
    return paths


def submit_condition(
    client, data: pd.DataFrame, condition: Condition, input_path: Path
) -> None:
    """Upload one condition's requests and save its manifest."""
    batch_ids = []
    with tempfile.TemporaryDirectory(prefix="cpc18_openai_batch_") as temp_name:
        input_files = _write_batch_input_files(data, condition, Path(temp_name))
        for chunk_number, path in enumerate(input_files, start=1):
            with path.open("rb") as input_file:
                uploaded = client.files.create(file=input_file, purpose="batch")
            client.files.wait_for_processing(
                uploaded.id, poll_interval=1.0, max_wait_seconds=300.0
            )
            time.sleep(FILE_PROPAGATION_WAIT_SECONDS)
            batch = client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"condition": condition.id[:512], "chunk": str(chunk_number)},
            )
            batch_ids.append(batch.id)
            print(
                f"  chunk {chunk_number}/{len(input_files)}: {batch.id} ({batch.status})",
                flush=True,
            )

    pricing = PRICING.get(condition.model)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "condition": condition.id,
        "model": condition.model,
        "effort": condition.effort,
        "instruction": condition.instruction,
        "instruction_text": condition.instruction_text,
        "input_csv": _stored_path(input_path),
        "input_csv_sha256": file_sha256(input_path),
        "row_count": len(data),
        "project": OPENAI_PROJECT,
        "organization": OPENAI_ORGANIZATION,
        "submitted_at": int(time.time()),
        "batch_ids": batch_ids,
        "pricing": pricing and {**pricing, "batch_multiplier": BATCH_PRICE_MULTIPLIER},
    }
    path = manifest_path(condition)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", help="default: experiment_config.MODELS")
    parser.add_argument("--efforts", nargs="+", help="default: experiment_config.EFFORTS")
    parser.add_argument(
        "--instructions", nargs="+", help="default: every key of INSTRUCTIONS"
    )
    parser.add_argument("--input", type=Path, default=INPUT_CSV, help="rows to predict")
    parser.add_argument(
        "--submit", action="store_true", help="really submit (billed); default is a dry run"
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.is_file():
        parser.error(
            f"Input CSV does not exist: {input_path}\n"
            "See the README for how to download the data and create a subset."
        )
    data = pd.read_csv(input_path).reset_index(drop=True)
    grid = conditions(args.models, args.efforts, args.instructions)
    pending = [c for c in grid if not manifest_path(c).exists()]

    for condition in grid:
        state = "to submit" if condition in pending else "already submitted, skipped"
        print(f"{condition.id}: {len(data):,} requests ({state})")
    print(
        f"{len(pending)} of {len(grid)} condition(s) to submit, "
        f"{len(pending) * len(data):,} requests in total."
    )

    if not args.submit:
        print("Dry run: nothing was sent. Add --submit to submit these batches.")
        return
    if not pending:
        return

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(Path(__file__).with_name(".env"), override=True)
    client = OpenAI(project=OPENAI_PROJECT, organization=OPENAI_ORGANIZATION)
    for condition in pending:
        print(f"Submitting {condition.id}", flush=True)
        submit_condition(client, data, condition, input_path)
    print("Submission is complete; this computer no longer needs to stay running.")


if __name__ == "__main__":
    main()
