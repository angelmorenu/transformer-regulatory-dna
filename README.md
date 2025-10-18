# Transformer Fine-Tuning for Regulatory DNA

**Author:** Angel Morenu  
**Course:** CAP 5510 – Bioinformatics  
**Instructor:** Dr. Tamer Kahveci  
**Due Date:** December 5, 2025

## Overview
This project evaluates transformer-based language models for DNA sequence analysis, focusing on:
1. Functional element classification (promoter/enhancer/DNase sites)  
2. Non-coding variant effect prediction via in-silico mutagenesis  

This work applies modern NLP methods to DNA sequences.

## Dataset Sources
- DeepSEA Training Bundle: https://deepsea.princeton.edu/help/
- ENCODE Portal: https://www.encodeproject.org/
- Roadmap Epigenomics Portal (WashU): https://egg2.wustl.edu/roadmap/web_portal/

## Models Implemented
- CNN Baselines: Basset (https://github.com/davek44/Basset), Basenji (https://github.com/calico/basenji)  
- Transformers: DNABERT-2 (https://huggingface.co/zhihan1996/DNABERT-2-117M), Nucleotide Transformer v2 (https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species)

## Evaluation Metrics
- AUROC, PR-AUC  
- Runtime / GPU memory profiling  
- Cross-cell-type transfer tests  
- Variant effect score correlation vs DeepSEA benchmarks  

## Environment Setup
```bash
conda create -n bio-transformers python=3.10 -y
conda activate bio-transformers
pip install -r requirements.txt
```
(Optional) If you prefer conda-only, use `environment.yml`.

## Reproducing Results
1. Download datasets → `data/raw/`  
2. Run `notebooks/01_data_preprocessing.ipynb`  
3. Train baselines → `notebooks/02_baseline_models.ipynb`  
4. Fine-tune transformers → `notebooks/03_transformer_finetuning.ipynb`  
5. Evaluate variant effects → `notebooks/04_variant_effects.ipynb`  
6. Visualize & export tables/figures → `notebooks/05_results_visualization.ipynb`

## Project Structure
```
transformer-regulatory-dna/
├── README.md
├── environment.yml
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_transformer_finetuning.ipynb
│   ├── 04_variant_effects.ipynb
│   └── 05_results_visualization.ipynb
├── src/
│   ├── data_utils.py
│   ├── train_baseline.py
│   ├── train_transformer.py
│   └── evaluate.py
├── results/
│   ├── metrics.csv
│   ├── plots/
│   └── logs/
└── report/
    └── Morenu_CAP5510_ProjectReport.tex
```

## Contact
angel.morenu@ufl.edu
