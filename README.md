# Transformer Fine-Tuning for Regulatory DNA

**Author:** Angel Morenu  
**Course:** CAP 5510 – Bioinformatics  
**Instructor:** Dr. Tamer Kahveci  
**Due Date:** December 5, 2025

---

## Overview
This project evaluates transformer-based sequence models for **functional genomics**—classifying regulatory elements (promoters, enhancers, DNase hypersensitive sites) and assessing **non-coding variant effects** using *in-silico saturation mutagenesis*.

It extends the DeepSEA benchmark with modern transformer backbones, comparing CNN baselines (Basset/Basenji) vs DNABERT-2 and the Nucleotide Transformer.

---

## Dataset Sources
- [DeepSEA Training Bundle](https://deepsea.princeton.edu/help/)
- [ENCODE Portal](https://www.encodeproject.org/)
- [Roadmap Epigenomics Portal (WashU)](https://egg2.wustl.edu/roadmap/web_portal/)

---

## Models Implemented
- **CNN Baselines:** Basset ([GitHub](https://github.com/davek44/Basset)), Basenji ([GitHub](https://github.com/calico/basenji))  
- **Transformers:**  
    - DNABERT-2 ([Hugging Face](https://huggingface.co/zhihan1996/DNABERT-2-117M))  
    - Nucleotide Transformer v2 ([Hugging Face](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species))

---

## Evaluation Metrics
- **Classification:** AUROC, PR-AUC  
- **Efficiency:** Runtime, GPU memory usage  
- **Transferability:** Cross-cell-type evaluation  
- **Variant Effects:** Correlation vs DeepSEA benchmark predictions

---

## Environment Setup
```bash
conda create -n bio-transformers python=3.10 -y
conda activate bio-transformers
pip install -r requirements.txt
# Alternatively, use environment.yml for a full conda environment
```

---

## Reproducing Results
1. Download datasets and place under data/raw/  
2. Preprocess data: notebooks/01_data_preprocessing.ipynb  
3. Train CNN baselines: notebooks/02_baseline_models.ipynb  
4. Fine-tune transformers: notebooks/03_transformer_finetuning.ipynb  
5. Run variant effect prediction: notebooks/04_variant_effects.ipynb  
6. Summarize & visualize results: notebooks/05_results_visualization.ipynb

---

## Report

- The full project report PDF is produced at the repository root as
    `Morenu_CAP5510_ProjectReport.pdf` (also generated from
    `Morenu_CAP5510_ProjectReport.tex`). To (re)build the PDF locally you can
    use:

```bash
latexmk -pdf -interaction=nonstopmode Morenu_CAP5510_ProjectReport.tex
# or: pdflatex Morenu_CAP5510_ProjectReport.tex (run twice)
```

If you don't have a local TeX toolchain, the repository's CI builds an
artifact on push to the `notebooks/visualization-fix` branch.

## Clean-up notes

- The repository ignores common macOS and editor cruft (see `gitignore.txt`).
- If you see files prefixed with `._` or `.DS_Store` in `git status`, you can
    safely remove them locally; they are ignored by default.

---

## Project Structure
```
transformer-regulatory-dna/
├── README.md
├── PROGRESS.md
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
│   ├── collate.py
│   ├── data_utils.py
│   ├── train_baseline.py
│   ├── train_transformer.py
│   └── evaluate.py
├── results/
│   ├── metrics.csv
│   ├── vep/
│   │   ├── deltas_test.npz
│   │   ├── deltas_test.tsv
│   │   ├── top50_test.tsv
│   │   └── summary.csv
│   ├── plots/
│   └── logs/
└── report/
        └── Morenu_CAP5510_ProjectReport.tex
```

---

## Contact
angel.morenu@ufl.edu  
University of Florida — CAP 5510 Bioinformatics
