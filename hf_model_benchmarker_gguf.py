# python -m pip install --upgrade pip
# pip install --upgrade \
#   setuptools wheel packaging scikit-build-core cmake ninja

# # Now build with CUDA
# CMAKE_ARGS="-DGGML_CUDA=on -DLLAVA_BUILD=off" \
#   pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
# pip install transformers accelerate
# pip install huggingface_hub tqdm

# python hf_model_benchmarker_gguf.py \
#   --prompts_file prompts.txt \
#   --models_file additional_models_to_test_08102025.txt \
#   --output_dir model_responses_gguf \
#   --checkpoint_file collection_checkpoint.json \
#   --hf_token \
#   --cache_dir ./models_gguf \
#   --n_ctx 8192 --max_new_tokens 5000 --threads 16 --gpu_layers -1

# -*- coding: utf-8 -*-
"""
hf_model_benchmarker_gguf.py

GGUF-only benchmarker that:
- reads prompts from a text file (one prompt per line)
- reads models from a text file (one per line, format: "repo_id|gguf_filename" or "repo_id")
- downloads required GGUF shards from Hugging Face (if not already present)
- runs generations using llama-cpp-python on NVIDIA (CUDA) via n_gpu_layers=-1
- writes per-model JSON outputs mirroring your original schema
- maintains a checkpoint and a streaming "raw_responses.txt"

Requirements (install on your NVIDIA box):
    pip install --upgrade --extra-index-url https://abetlen.github.io/llama-cpp-python/ llama-cpp-python
    pip install huggingface_hub tqdm

Notes:
- If a model line only has "repo_id" and no filename, the script will try to pick a GGUF file
  by preferring higher-precision quants (Q6_K > Q5_K > Q4_K ...), otherwise the largest file.
- If the filename points to a shard (e.g., ...-00001-of-00003.gguf or .part1of2), all shards
  with the same base stem will be downloaded; llama.cpp will auto-load them.

Usage example:
    python hf_model_benchmarker_gguf.py \
        --prompts_file prompts.txt \
        --models_file additional_models_to_test_08102025.txt \
        --output_dir model_responses_gguf \
        --checkpoint_file collection_checkpoint.json \
        --hf_token YOUR_HF_TOKEN \
        --cache_dir ./models_gguf \
        --n_ctx 8192 --max_new_tokens 1024 --threads 16
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
import argparse
import traceback

from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from llama_cpp import Llama

# --------------------------------
# Helpers for GGUF discovery
# --------------------------------

_SHARD_PATTERNS = [
    (re.compile(r"-\d{5}-of-\d{5}\.gguf$"), ""),           # -00001-of-00003.gguf -> .gguf
    (re.compile(r"\.part\d+of\d+$"), ""),                 # .part1of2 -> remove suffix
]

def _gguf_stem(fname: str) -> str:
    stem = fname
    for pat, repl in _SHARD_PATTERNS:
        stem = pat.sub(repl, stem)
    return stem

def _quant_rank(filename: str) -> int:
    """Heuristic ranking: prefer higher precision quants when auto-selecting."""
    name = filename.lower()
    # Higher is better
    order = ["q8", "q6_k", "q6", "q5_k", "q5", "q4_k_m", "q4_k_s", "q4_k", "q4", "q3", "q2"]
    for i, tok in enumerate(order[::-1]):
        if tok in name:
            # return increasing rank with better quant -> smaller number is better
            return i
    # unknown quant -> worst
    return len(order)

def _list_gguf(repo_id: str):
    files = list_repo_files(repo_id)
    return [f for f in files if f.lower().endswith(".gguf")]

def _pick_best_gguf(ggufs):
    """Choose a GGUF file: prefer higher-precision quant; fallback to largest filename lex order."""
    if not ggufs: 
        return None
    # rank by quant then by name length as fallback
    ggufs_sorted = sorted(ggufs, key=lambda f: (_quant_rank(f), -len(f)))
    return ggufs_sorted[0]

def _collect_shards_for(fname: str, all_files):
    """Return all shard filenames that belong to the same base stem as fname (including fname)."""
    base = _gguf_stem(fname)
    shards = []
    for f in all_files:
        if not f.lower().endswith(".gguf"):
            continue
        if f == base or f.startswith(base.replace(".gguf","")):
            shards.append(f)
    # Ensure deterministic order
    return sorted(set(shards))

# --------------------------------
# Main collector
# --------------------------------

class GGUFResponseCollector:
    def __init__(self, prompts_file, models_file, output_dir, checkpoint_file,
                 hf_token=None, cache_dir="models_gguf",
                 n_ctx=8192, max_new_tokens=1024, threads=8, gpu_layers=-1, top_p=0.95, temperature=0.7):
        self.prompts_file = prompts_file
        self.models_file = models_file
        self.output_dir = Path(output_dir)
        self.checkpoint_file = self.output_dir / checkpoint_file
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.hf_token = hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.n_ctx = int(n_ctx)
        self.max_new_tokens = int(max_new_tokens)
        self.threads = int(threads)
        self.gpu_layers = int(gpu_layers)
        self.top_p = float(top_p)
        self.temperature = float(temperature)

        self.prompts = self._load_prompts()
        self.models = self._load_models()

        self.checkpoint = self._load_checkpoint()

        # Thinking tags (for parsing)
        self.thinking_tags = [
            ("<think>", "</think>"),
            ("<thinking>", "</thinking>"),
            ("<|thinking|>", "<|/thinking|>"),
            ("<thought>", "</thought>"),
            ("<|start_thinking|>", "<|end_thinking|>")
        ]

    def _load_prompts(self):
        with open(self.prompts_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts")
        return prompts

    def _load_models(self):
        """Read models list file: one per line, either 'repo_id|gguf_filename' or just 'repo_id'."""
        items = []
        for line in Path(self.models_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                repo_id, fname = [x.strip() for x in line.split("|", 1)]
            else:
                repo_id, fname = line, None
            items.append((repo_id, fname))
        print(f"Loaded {len(items)} models to evaluate")
        return items

    def _load_checkpoint(self):
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            print(f"Loaded checkpoint. Completed models: {len(ckpt.get('completed_models', []))}")
            return ckpt
        return {"completed_models": [], "partial_results": {}}

    def _save_checkpoint(self):
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(self.checkpoint, f, indent=2)

    def _parse_thinking_response(self, response: str):
        thinking_content = None
        final_response = response
        for start_tag, end_tag in self.thinking_tags:
            if end_tag in response:
                thinking_content = response.split(end_tag)[0].replace(start_tag, "").strip()
                final_response = response.split(end_tag)[1].strip()
                break
        return {
            "full_response": response,
            "thinking": thinking_content,
            "final_response": final_response,
            "has_thinking": thinking_content is not None
        }

    # ---- HF downloads & local path resolution ----
    def _ensure_local_model(self, repo_id: str, fname_hint: str | None):
        """Download needed gguf file(s) into cache_dir/repo__name/ and return path to main file."""
        api = HfApi(token=self.hf_token)
        all_gguf = _list_gguf(repo_id)
        if not all_gguf:
            raise RuntimeError(f"No GGUF files found in {repo_id}")

        if fname_hint:
            # Ensure hint exists
            if fname_hint not in all_gguf:
                # try locate by basename match
                matches = [f for f in all_gguf if f.endswith(Path(fname_hint).name)]
                if not matches:
                    raise RuntimeError(f"GGUF file '{fname_hint}' not found in {repo_id}. Available: {all_gguf[:10]}...")
                fname = matches[0]
            else:
                fname = fname_hint
        else:
            fname = _pick_best_gguf(all_gguf)

        shards = _collect_shards_for(fname, all_gguf)
        local_dir = self.cache_dir / repo_id.replace("/", "__")
        local_dir.mkdir(parents=True, exist_ok=True)

        for f in shards:
            print(f"[DL] {repo_id} / {f}")
            hf_hub_download(repo_id, f, local_dir=local_dir, local_dir_use_symlinks=False, token=self.hf_token)

        # Return path to main file (first shard or base)
        main_file = local_dir / fname
        if not main_file.exists():
            # If we selected a base but HF gives shards only, pick the first shard we downloaded
            shard_paths = sorted(list(local_dir.glob(_gguf_stem(fname).replace(".gguf", "") + "*")))
            if shard_paths:
                main_file = shard_paths[0]
            else:
                # fallback: pick any gguf in dir
                any_gguf = sorted(local_dir.glob("*.gguf"))
                if not any_gguf:
                    raise RuntimeError(f"Downloaded no GGUF files for {repo_id}")
                main_file = any_gguf[0]

        return str(main_file)

    # ---- llama.cpp inference ----
    def _open_llama(self, model_path: str) -> Llama:
        print(f"Loading GGUF with llama-cpp: {model_path}")
        llm = Llama(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.gpu_layers,  # -1 to offload all possible
            n_threads=self.threads,
            # You can tune batch size & rope scaling via additional kwargs if desired
        )
        return llm

    def _generate_response_llama(self, llm: Llama, prompt: str, prompt_idx: int):
        try:
            # Simple single-turn prompt. If your model uses chat, you could prepend system/user markers.
            output = llm(
                prompt,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stop=["</s>", "###", "User:", "Assistant:"],
                echo=False
            )
            response = output["choices"][0]["text"].strip()
            parsed = self._parse_thinking_response(response)
            return {
                "prompt_idx": prompt_idx,
                "prompt": prompt,
                "raw_response": response,
                "parsed_response": parsed,
                "response_length": len(response),
                "timestamp": datetime.now().isoformat(),
                "error": None
            }
        except Exception as e:
            return {
                "prompt_idx": prompt_idx,
                "prompt": prompt,
                "raw_response": None,
                "parsed_response": None,
                "response_length": 0,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def _save_model_responses(self, model_label: str, responses):
        safe_name = model_label.replace("/", "_").replace("|", "_")
        out_file = self.output_dir / f"{safe_name}_responses.json"
        data = {
            "model": model_label,
            "timestamp": datetime.now().isoformat(),
            "device": "cuda",  # llama-cpp will use CUDA when built with cuBLAS; otherwise CPU
            "total_prompts": len(self.prompts),
            "prompts": self.prompts,
            "responses": responses
        }
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved responses to {out_file}")

    def _collect_model(self, repo_id: str, gguf_filename: str | None):
        label = repo_id if gguf_filename is None else f"{repo_id}|{gguf_filename}"
        print("\n" + "="*60)
        print(f"Collecting responses from: {label}")
        print("="*60)
        responses = []
        try:
            model_path = self._ensure_local_model(repo_id, gguf_filename)
            llm = self._open_llama(model_path)

            for i, prompt in enumerate(tqdm(self.prompts, desc="Processing prompts")):
                res = self._generate_response_llama(llm, prompt, i)
                responses.append(res)

                # Append raw line for durability
                safe_text = (res['raw_response'] or 'ERROR').replace("\n", " ")
                line = f"|||{label}|{i}|{safe_text}\n"
                # line = f"|||{label}|{i}|{(res['raw_response'] or 'ERROR').replace('\n',' ')}\n"
                with open(self.output_dir / "raw_responses.txt", "a", encoding="utf-8") as f:
                    f.write(line)

                # Periodic checkpoint
                if (i + 1) % 5 == 0:
                    self.checkpoint["partial_results"][label] = responses
                    self._save_checkpoint()

            # save JSON
            self._save_model_responses(label, responses)

            # update checkpoint
            self.checkpoint.setdefault("completed_models", [])
            self.checkpoint["completed_models"].append(label)
            if label in self.checkpoint.get("partial_results", {}):
                del self.checkpoint["partial_results"][label]
            self._save_checkpoint()

        except Exception as e:
            print(f"Error processing {label}: {e}")
            traceback.print_exc()
            if responses:
                self.checkpoint["partial_results"][label] = responses
                self._save_checkpoint()

    def run_collection(self):
        print(f"\nStarting GGUF response collection at {datetime.now()}")
        print(f"Total models to process: {len(self.models)}")
        print(f"Already completed: {len(self.checkpoint.get('completed_models', []))}")
        for idx, (repo_id, gguf_fname) in enumerate(self.models, start=1):
            label = repo_id if gguf_fname is None else f"{repo_id}|{gguf_fname}"
            if label in self.checkpoint.get("completed_models", []):
                print(f"[{idx}/{len(self.models)}] Skipping (completed): {label}")
                continue
            if label in self.checkpoint.get("partial_results", {}):
                print(f"[{idx}/{len(self.models)}] Found partial results for {label} (will restart)")
            print(f"[{idx}/{len(self.models)}] Processing: {label}")
            self._collect_model(repo_id, gguf_fname)
        print("\n" + "="*60)
        print("Collection complete.")
        print(f"Outputs: {self.output_dir}")

def main():
    ap = argparse.ArgumentParser(description="Run GGUF response collection with llama-cpp-python (CUDA).")
    ap.add_argument("--prompts_file", default="prompts.txt", help="Path to prompts file")
    ap.add_argument("--models_file", default="models.txt", help="Path to models list (repo|filename per line)")
    ap.add_argument("--output_dir", default="model_responses_gguf", help="Output directory")
    ap.add_argument("--checkpoint_file", default="collection_checkpoint.json", help="Checkpoint filename (in output dir)")
    ap.add_argument("--hf_token", default=None, help="Hugging Face access token")
    ap.add_argument("--cache_dir", default="models_gguf", help="Local cache dir for GGUF downloads")
    ap.add_argument("--n_ctx", type=int, default=8192, help="Context window")
    ap.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens to generate per prompt")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads for llama.cpp")
    ap.add_argument("--gpu_layers", type=int, default=-1, help="Layers to offload to GPU (-1 = all possible)")
    ap.add_argument("--top_p", type=float, default=0.95, help="top_p sampling")
    ap.add_argument("--temperature", type=float, default=0.7, help="temperature")
    args = ap.parse_args()

    collector = GGUFResponseCollector(
        prompts_file=args.prompts_file,
        models_file=args.models_file,
        output_dir=args.output_dir,
        checkpoint_file=args.checkpoint_file,
        hf_token=args.hf_token,
        cache_dir=args.cache_dir,
        n_ctx=args.n_ctx,
        max_new_tokens=args.max_new_tokens,
        threads=args.threads,
        gpu_layers=args.gpu_layers,
        top_p=args.top_p,
        temperature=args.temperature
    )
    collector.run_collection()

if __name__ == "__main__":
    main()
    