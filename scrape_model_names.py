import csv
import os
import time
import argparse
from datetime import datetime
from huggingface_hub import HfApi

def scrape_and_save(hf_token, search_terms, rate_limit=0.01, resume_timestamp=None):
    api = HfApi(token=hf_token)

    # Use given timestamp or create a new one
    timestamp = resume_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"hf_models_{timestamp}.csv"
    checkpoint_file = f"hf_models_checkpoint_{timestamp}.txt"

    processed_terms = set()
    if resume_timestamp and os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            processed_terms = {line.strip() for line in f if line.strip()}

    mode = "a" if os.path.exists(output_file) else "w"
    with open(output_file, mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if mode == "w":
            writer.writerow(["repo", "search_term"])  # header

        for term in search_terms:
            if term in processed_terms:
                print(f"Skipping already processed term: {term}")
                continue

            print(f"Searching for: {term}")
            try:
                # Get ALL results (no arbitrary cutoff)
                results = api.list_models(search=term, full=True)
                for repo in results:
                    writer.writerow([repo.modelId, term])
            except Exception as e:
                print(f"Error with term '{term}': {e}")

            # mark term as processed
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                f.write(term + "\n")

            csvfile.flush()
            time.sleep(rate_limit)

    print(f"Done. Results saved in {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token", required=True, help="Hugging Face API token")
    parser.add_argument("--terms", required=True, help="Search terms in line separated text file")
    parser.add_argument("--rate", type=float, default=0.01, help="Rate limit delay (s)")
    parser.add_argument("--resume-timestamp", help="Timestamp of a previous run to resume from")
    args = parser.parse_args()

    search_term_file = args.terms
    with open(search_term_file, 'r') as f:
        search_terms = [x.strip() for x in f.readlines()]

    scrape_and_save(args.hf_token, search_terms, rate_limit=args.rate, resume_timestamp=args.resume_timestamp)
