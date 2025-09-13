# pip install transformers, accelerate
# pip install -U jinja2
# pip install -U torch

import os
import json
import torch
import gc
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import traceback
import sys
import re
import argparse
from accelerate import infer_auto_device_map, dispatch_model

# define a constant for HF_ACCESS_TOKEN

# # Output JSON structure for each model
# {
#   "model": "model/name",
#   "timestamp": "...",
#   "device": "cuda",
#   "total_prompts": 50,
#   "prompts": ["prompt1", "prompt2", ...],
#   "responses": [
#     {
#       "prompt_idx": 0,
#       "prompt": "full prompt text",
#       "raw_response": "full model response",
#       "parsed_response": {
#         "full_response": "...",
#         "thinking": "thinking content if present",
#         "final_response": "response without thinking",
#         "has_thinking": true/false
#       },
#       "response_length": 1234,
#       "timestamp": "...",
#       "error": null
#     }
#   ]
# }

# # Raw Response File format
# |||<model_index>|<prompt_index>|<raw_response_text>


class ResponseCollector:
    def __init__(self, prompts_file="prompts.txt", models_file="models.txt", 
                 output_dir="model_responses", checkpoint_file="collection_checkpoint.json"):
        self.prompts_file = prompts_file
        self.models_file = models_file
        self.output_dir = Path(output_dir)
        self.checkpoint_file = self.output_dir / checkpoint_file
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Load prompts and models
        self.prompts = self._load_prompts()
        self.models = self._load_models()
        
        # Load checkpoint if exists
        self.checkpoint = self._load_checkpoint()
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Common thinking model tags
        self.thinking_tags = [
            ("<think>", "</think>"),
            ("<thinking>", "</thinking>"),
            ("<|thinking|>", "<|/thinking|>"),
            ("<thought>", "</thought>"),
            ("<|start_thinking|>", "<|end_thinking|>")
        ]
        
    def _load_prompts(self):
        """Load prompts from file"""
        with open(self.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts")
        return prompts
    
    def _load_models(self):
        """Load model names from file"""
        with open(self.models_file, 'r') as f:
            models = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(models)} models to evaluate")
        return models
    
    def _load_checkpoint(self):
        """Load checkpoint to resume from crashes"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"Loaded checkpoint. Completed models: {len(checkpoint['completed_models'])}")
            return checkpoint
        else:
            return {"completed_models": [], "partial_results": {}}
    
    def _save_checkpoint(self):
        """Save current progress"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def _parse_thinking_response(self, response):
        """Extract thinking and final response from thinking models"""
        thinking_content = None
        final_response = response
        
        # Try to find thinking tags
        for start_tag, end_tag in self.thinking_tags:
            if end_tag in response:
                thinking_content = response.split(end_tag)[0].replace(start_tag,'').strip()
                actual_response = response.split(end_tag)[1].strip()
                break
                
        
        return {
            "full_response": response,
            "thinking": thinking_content,
            "final_response": final_response,
            "has_thinking": thinking_content is not None
        }
    
    def _save_model_responses(self, model_name, responses):
        """Save responses for a single model"""
        safe_model_name = model_name.replace("/", "_")
        output_file = self.output_dir / f"{safe_model_name}_responses.json"
        
        # Create comprehensive output
        output_data = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "device": self.device,
            "total_prompts": len(self.prompts),
            "prompts": self.prompts,  # Include all prompts for reference
            "responses": responses
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved responses for {model_name} to {output_file}")
    
    def _generate_response(self, model, tokenizer, prompt, prompt_idx, max_new_tokens=5000):
        """Generate response from model with high token limit"""
        try:
            # Apply chat template if available
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt}]
                input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                input_text = prompt
            
            # Tokenize with truncation for very long prompts
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_new_tokens)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with high token limit
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    # use_cache=True,
                    # repetition_penalty=1.1,  # Add this to prevent loops
                )
            
            # Decode full response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove input prompt)
            input_text_decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
            if full_response.startswith(input_text_decoded):
                response = full_response[len(input_text_decoded):].strip()
            else:
                # Fallback: use the portion after the input length
                response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Parse thinking models
            parsed_response = self._parse_thinking_response(response)
            
            return {
                "prompt_idx": prompt_idx,
                "prompt": prompt,
                "raw_response": response,
                "parsed_response": parsed_response,
                "response_length": len(response),
                "timestamp": datetime.now().isoformat(),
                "error": None
            }
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                "prompt_idx": prompt_idx,
                "prompt": prompt,
                "raw_response": None,
                "parsed_response": None,
                "response_length": 0,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    # def _generate_response_llama(self, model, tokenizer, prompt, prompt_idx, max_new_tokens=5000):
    #     """
    #     Generate response from a GGUF model using llama-cpp-python, applying chat template from metadata.
    #     """
    #     try:
    #         # Wrap prompt in a single-turn chat message structure
    #         messages = [
    #             {"role": "user", "content": prompt}
    #         ]
    
    #         # Use model's own embedded chat template for formatting
    #         formatted_prompt = model.apply_chat_template(messages, add_generation_prompt=True)
    
    #         output = model(
    #             formatted_prompt,
    #             max_tokens=max_new_tokens,
    #             temperature=0.7,
    #             top_p=0.95,
    #             stop=["</s>", "###", "User:", "Assistant:"],
    #             echo=False
    #         )
    
    #         response = output["choices"][0]["text"].strip()
    #         parsed_response = self._parse_thinking_response(response)
    
    #         return {
    #             "prompt_idx": prompt_idx,
    #             "prompt": prompt,
    #             "raw_response": response,
    #             "parsed_response": parsed_response,
    #             "response_length": len(response),
    #             "timestamp": datetime.now().isoformat(),
    #             "error": None
    #         }
    
    #     except Exception as e:
    #         print(f"Error generating response (GGUF): {e}")
    #         return {
    #             "prompt_idx": prompt_idx,
    #             "prompt": prompt,
    #             "raw_response": None,
    #             "parsed_response": None,
    #             "response_length": 0,
    #             "timestamp": datetime.now().isoformat(),
    #             "error": str(e)
    #         }
    
    def _collect_model_responses(self, model_name):
        """Collect responses from a single model for all prompts"""
        print(f"\n{'='*60}")
        print(f"Collecting responses from: {model_name}")
        print(f"{'='*60}")
        
        responses = []
        
        try:
            # Load model and tokenizer
            print(f"Loading model and tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=HF_ACCESS_TOKEN)
            
            # Set padding token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with appropriate settings
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                gguf_file=gguf_filename,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                token=HF_ACCESS_TOKEN,
            )
        
            # if self.device == "cpu":
            #     model = model.to(self.device)
            # if torch.cuda.device_count() > 1:
            #     max_memory = {i: "70GiB" for i in range(torch.cuda.device_count())}
            #     device_map = infer_auto_device_map(model, max_memory=max_memory)
            #     model = dispatch_model(model, device_map=device_map)
            # else:
            #     model = model.to("cuda")  # single GPU fallback
    
            model.eval()
            print(f"Model loaded successfully on {self.device}")
            
           
            # Process each prompt
            for i, prompt in enumerate(tqdm(self.prompts, desc="Processing prompts")):
                print(f"\nPrompt {i+1}/{len(self.prompts)}: {prompt[:80]}...")

                # if isinstance(model, Llama):
                #     response_data = self._generate_response_llama(model, prompt, i)
                # else:
                #     response_data = self._generate_response(model, tokenizer, prompt, i)
                response_data = self._generate_response(model, tokenizer, prompt, i)
                responses.append(response_data)

                # Save line-wise robust output
                safe_model_idx = self.models.index(model_name)
                line = f"|||{safe_model_idx}|{i}|{response_data['raw_response'] or 'ERROR'}\n"
                with open(self.output_dir / "raw_responses.txt", "a", encoding="utf-8") as f:
                    f.write(line)
                
                # Print summary
                if response_data['error']:
                    print(f"  ERROR: {response_data['error']}")
                else:
                    print(f"  Response length: {response_data['response_length']} chars")
                    if response_data['parsed_response']['has_thinking']:
                        print(f"  Contains thinking section")
                
                # Save partial results periodically
                if (i + 1) % 5 == 0:
                    self.checkpoint["partial_results"][model_name] = responses
                    self._save_checkpoint()
                    print(f"  Checkpoint saved at prompt {i+1}")
            
            # Clean up model from memory
            # del model
            # del tokenizer
            # gc.collect()
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            
            return responses
            
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            traceback.print_exc()
            
            # Save any partial results
            if responses:
                self.checkpoint["partial_results"][model_name] = responses
                self._save_checkpoint()
            
            return None
    
    def run_collection(self):
        """Run response collection on all models"""
        print(f"\nStarting response collection at {datetime.now()}")
        print(f"Total models to process: {len(self.models)}")
        print(f"Already completed: {len(self.checkpoint['completed_models'])}")
        
        for model_idx, model_name in enumerate(self.models):
            print(f"\n[{model_idx+1}/{len(self.models)}] Processing {model_name}")
            
            # Skip if already completed
            if model_name in self.checkpoint['completed_models']:
                print(f"Skipping {model_name} (already completed)")
                continue
            
            # Check for partial results
            if model_name in self.checkpoint['partial_results']:
                print(f"Found {len(self.checkpoint['partial_results'][model_name])} partial results for {model_name}")
                # For simplicity, we'll restart this model
                # You could implement proper resume logic here
            
            # Collect responses
            responses = self._collect_model_responses(model_name)
            
            if responses:
                # Save responses
                self._save_model_responses(model_name, responses)
                
                # Update checkpoint
                self.checkpoint['completed_models'].append(model_name)
                if model_name in self.checkpoint['partial_results']:
                    del self.checkpoint['partial_results'][model_name]
                self._save_checkpoint()
                
                print(f"\nCompleted {model_name}")
                print(f"Progress: {len(self.checkpoint['completed_models'])}/{len(self.models)} models")
            else:
                print(f"\nFailed to process {model_name}")
        
        print(f"\n{'='*60}")
        print(f"Collection complete!")
        print(f"Successfully processed: {len(self.checkpoint['completed_models'])}/{len(self.models)} models")
        print(f"Results saved in: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run HF model response collection.")
    parser.add_argument("--prompts_file", default="prompts.txt", help="Path to the prompts file (default: prompts.txt)")
    parser.add_argument("--models_file", default="models.txt", help="Path to the models file (default: models.txt)")
    parser.add_argument("--output_dir", default="model_responses", help="Directory to store output files")
    parser.add_argument("--checkpoint_file", default="collection_checkpoint.json", help="Checkpoint file name")
    
    args = parser.parse_args()

    # Check if input files exist
    if not os.path.exists(args.prompts_file):
        print(f"Error: {args.prompts_file} not found!")
        sys.exit(1)

    if not os.path.exists(args.models_file):
        print(f"Error: {args.models_file} not found!")
        sys.exit(1)

    # Run collection
    collector = ResponseCollector(
        prompts_file=args.prompts_file,
        models_file=args.models_file,
        output_dir=args.output_dir,
        checkpoint_file=args.checkpoint_file
    )
    collector.run_collection()

if __name__ == "__main__":
    main()
    