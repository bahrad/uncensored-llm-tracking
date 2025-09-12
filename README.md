# Uncensored AI in the Wild: Tracking Publicly Available and Locally Deployable LLMs

This repository contains the code and data for the research paper "Uncensored AI in the Wild: Tracking Publicly Available and Locally Deployable LLMs" which presents the first large-scale empirical analysis of safety-modified open-weight language models.

## Overview

This study analyzes model retrieved using search terms from Hugging Face to identify models explicitly adapted to bypass alignment safeguards. The research demonstrates systematic patterns in how these "uncensored" models are created, distributed, and optimized for local deployment.

## Repository Structure

### Data Collection Scripts
- `scrape_model_names.py` - Main scraper for (incrementally) retrieving model names that hit safety/uncensorship keywords from Hugging Face
- `scrape_metadata.py` - Script to retrieve JSON metadata for model repositories identified using the search script.
- `get_repo_download_counts.py` - Script for obtaining total downloads for all time up to cutoff date for a list of model repos (model IDs)

### Analysis Scripts and Notebooks  
- `process_scrape_results.ipynb` - Filter and process raw scrape data and generates normalized datasets with family attribution
- `hf_model_benchmarker_.py` - Evaluates selected models using Hugging Face API
- `hf_model_benchmarker_gguf.py` - Evaluates GGUF-format models using llama.cpp
- `unsafe_prompt_evaluation.ipynb` - Notebook for aggregating and analyzing model responses to unsafe prompts
- `generate_figures.ipynb` - Generates the paper figures and tables from the output of `process_scrape_results.ipynb` and benchmarking

### Generated Data Files

#### Model Scraping
- `safety_terms.txt` - Search terms used to retrieve models from Hugging Face
- `model_list.txt` - List of model repository names retreived from Hugging Face
- `repo_catalog.tsv` - Complete catalog of scraped repositories with metadata after processing with `model_trends_analysis.py`
- `evaluated_models_metadata.csv` - Metadata for the subset of models evaluated for safety

#### Model Evaluation
- `modified_model_evaluation_revised.csv` - Safety evaluation results for tested models
- `evaluated_models_metadata_revised` - Metadata for tested models
- `prompt_list.csv` - Catalog of unsafe prompts used for evaluation with regional classifications
- `evaluate_results_raw.json` - Raw results (including full responses) from evaluation experiments (WARNING: DATA MAY CONTAIN UNSAFE MATERIAL)

## Ethics and Safety

This research examines publicly available models to understand AI safety challenges. The model responses for safety evaluation are not included in this public repository due to their sensitive nature, but are available from the author by request for legitimate research purposes.

## Citation

```bibtex
[PLACEHOLDER - Citation to be added upon publication]
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the research or data access requests, please contact the author.
