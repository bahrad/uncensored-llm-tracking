#!/usr/bin/env python3
import argparse
import csv
import os
import time
import random

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

# https://huggingface.co/api/models/<username>/<repo>?expand%5B%5D=downloads&expand%5B%5D=downloadsAllTime


def fetch_repo_downloads(api: HfApi, repo_id: str, max_retries: int = 10, kind="model"):
    """
    Returns (monthly_downloads, total_downloads) for a given repo_id.
    Retries on transient errors with exponential backoff.
    """
    attempt = 0
    while True:
        try:
            if kind == 'model':
                info = api.model_info(repo_id, expand=["downloads","downloadsAllTime"])
            else:
                info = api.dataset_info(repo_id, expand=["downloads","downloadsAllTime"])
            monthly = getattr(info, "downloads", None)
            total = getattr(info, "downloads_all_time", None)
            return monthly, total
        except HfHubHTTPError as e:
            attempt += 1
            if attempt > max_retries:
                return None, None
            base = 1.0 if e.response is not None and e.response.status_code == 429 else 0.5
            sleep_s = base * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            time.sleep(sleep_s)
        except Exception:
            attempt += 1
            if attempt > max_retries:
                return None, None
            sleep_s = 0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            time.sleep(sleep_s)


def main():
    parser = argparse.ArgumentParser(description="Fetch HF repo downloads to CSV.")
    parser.add_argument("input_txt", help="Path to text file with repo IDs (one per line)")
    parser.add_argument("output_csv", help="Path to write CSV output")
    parser.add_argument("--hf-token", help="HF token (else uses env HF_TOKEN if set)", default=None)
    parser.add_argument("--pause", type=float, default=0.05,
                        help="Pause between successful calls (seconds). Default=0.05")
    parser.add_argument("--checkpoint", type=int, default=100,
                        help="How often to flush/sync the file. Default=100")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", action="store_true", help="Download model metadata")
    group.add_argument("--dataset", action="store_true", help="Download dataset metadata")
    args = parser.parse_args()

    api = HfApi(token=args.hf_token) if args.hf_token else HfApi()

    kind = "model" if args.model else "dataset"

    with open(args.input_txt, "r", encoding="utf-8") as f:
        repo_ids = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    # open once, write as we go
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_id", "monthly_downloads", "total_downloads"])

        for i, repo_id in enumerate(repo_ids, 1):
            monthly, total = fetch_repo_downloads(api, repo_id, kind=kind)
            writer.writerow([
                repo_id,
                "NaN" if monthly is None else monthly,
                "NaN" if total is None else total,
            ])

            if i % args.checkpoint == 0:
                f.flush()
                os.fsync(f.fileno())
                print(f"Checkpoint: {i}/{len(repo_ids)} repos written...", flush=True)

            time.sleep(args.pause)

    print(f"Done. Wrote {len(repo_ids)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
