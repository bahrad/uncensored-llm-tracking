import json
import time
import argparse
from datetime import datetime
from huggingface_hub import HfApi

# Exemplary output
# {"id": "facebook/opt-125m", "metadata": {"modelId": "facebook/opt-125m", "downloads": 12345}}

# # How to get metadata manually instead of using the scraper code below
# # model
# GET https://huggingface.co/api/models/{namespace}/{model}
# # example
# https://huggingface.co/api/models/facebook/opt-125m
# # private repo
# curl -H "Authorization: Bearer $HF_TOKEN" \
#   https://huggingface.co/api/models/facebook/opt-125m
# # dataset
# GET https://huggingface.co/api/datasets/{namespace}/{dataset}
# # example
# https://huggingface.co/api/datasets/allenai/c4
# # private repo
# curl -H "Authorization: Bearer $HF_TOKEN" \
#   https://huggingface.co/api/datasets/allenai/c4


def download_metadata(hf_token, ids, rate_limit=0.01, kind="model"):
    api = HfApi(token=hf_token)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if kind == "model":
        output_file = f"hf_model_metadata_{timestamp}.jsonl"
    else:
        output_file = f"hf_dataset_metadata_{timestamp}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for item_id in ids:
            print(f"Fetching: {item_id}")
            record = {"id": item_id}       # to output an embedded JSON
            try:
                if "/" not in item_id:
                    raise ValueError("Invalid ID, must be namespace/name")

                if kind == "model":
                    meta = api.model_info(item_id)
                else:
                    meta = api.dataset_info(item_id)

                record["metadata"] = meta.__dict__

            except Exception as e:
                record["metadata"] = "ERROR"
                print(f"Error fetching {item_id}: {e}")

            f.write(json.dumps(record, default=str) + "\n")       # to output an embedded JSON
            f.flush()
            time.sleep(rate_limit)

    print(f"Done. Metadata saved in {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token", required=True, help="Hugging Face API token")
    parser.add_argument("--ids", required=True, help="Line-separated file listing namespace/model or namespace/dataset IDs")
    parser.add_argument("--rate", type=float, default=0.01, help="Rate limit delay (s)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", action="store_true", help="Download model metadata")
    group.add_argument("--dataset", action="store_true", help="Download dataset metadata")
    args = parser.parse_args()

    with open(args.ids, 'r') as f:
        ids = [x.strip() for x in f.readlines()]

    kind = "model" if args.model else "dataset"
    download_metadata(args.hf_token, ids, rate_limit=args.rate, kind=kind)