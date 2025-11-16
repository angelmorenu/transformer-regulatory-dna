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
        # Transformer Fine-Tuning for Regulatory DNA

        **Author:** Angel Morenu

        This repository contains a reproducible pipeline for evaluating transformer-based models on regulatory DNA classification and variant-effect prediction. It includes data preprocessing, CNN baselines, transformer linear-probe experiments, variant-effect (VEP) analysis, plotting, and a LaTeX report with auto-inserted numeric summaries comparing our variant-effect scores to DeepSEA.

        Highlights
        - Transformer probe (frozen encoder + linear head) experiments
        - CNN baselines (Basset-style) for direct comparison
        - Variant effect prediction and Top-K enrichment vs DeepSEA
        - Reproducible notebooks and scripts that generate plots under `notebooks/results/`
        - Project report: `Morenu_CAP5510_ProjectReport.tex` and tracked `Morenu_CAP5510_ProjectReport.pdf`

        ## Quick setup
        Use the provided environment files to recreate the Python environment:

        ```bash
        conda env create -f environment.yml   # preferred (if using conda)
        conda activate bio-transformers
        pip install -r requirements.txt       # alternative
        ```

        ## How to reproduce main artifacts
        - Preprocess data: run `notebooks/01_data_preprocessing.ipynb` (or `python src/collate.py`)
        - Train baselines: `notebooks/02_baseline_models.ipynb`
        - Fine-tune / probe: `notebooks/03_transformer_finetuning.ipynb` or `python src/train_transformer.py`
        - Variant effects: `notebooks/04_variant_effects.ipynb` (VEP deltas and top-K tables)
        - Visualizations and summaries (generate plots in-place): `notebooks/05_results_visualization.ipynb`

        The notebooks write results and plots to `notebooks/results/plots/` and VEP outputs to `notebooks/results/vep/`.

        ## Report (LaTeX)
        - The project report lives in `Morenu_CAP5510_ProjectReport.tex` and the generated PDF is tracked at the repository root `Morenu_CAP5510_ProjectReport.pdf`.
        - To build locally (requires a TeX Live / MacTeX installation):

        ```bash
        cd "$(pwd)"
        latexmk -pdf -interaction=nonstopmode Morenu_CAP5510_ProjectReport.tex
        # or: pdflatex Morenu_CAP5510_ProjectReport.tex  # run twice
        ```

        - The report includes an auto-generated macros file `notebooks/results/plots/vep_deepsea_summary.tex` (produced by `scripts/write_vep_deepsea_tex.py`) which injects numeric summaries (Spearman, top-K overlaps) into the LaTeX source. If you regenerate VEP comparisons, re-run that script before rebuilding the report.

        ## Useful scripts and locations
        - `src/compute_vep_vs_deepsea.py` — align VEP and DeepSEA per-position scores and compute Spearman / top-K enrichments.
        - `scripts/write_vep_deepsea_tex.py` — write LaTeX macros consumed by the report (`vep_deepsea_summary.tex`).
        - `notebooks/results/plots/` — plots and auto-generated LaTeX macros used by the report.
        - `notebooks/results/vep/` — VEP outputs (deltas, top-K lists).

        ## Tests
        - Run unit tests (pytest):

        ```bash
        pytest -q
        ```

        ## Clean-up and housekeeping
        - The repository `.gitignore` now ignores common macOS/editor cruft (`.DS_Store`, `._*`) and large artifact files while explicitly allowing the main report PDF `Morenu_CAP5510_ProjectReport.pdf` to be tracked.
        - If you see stray `._` files in `git status`, they are macOS resource forks and can be safely removed locally; they are ignored by git after updating `.gitignore`.

        ## Branches / CI
        - The branch `notebooks/visualization-fix` contains fixes to the LaTeX macros and an included report PDF; CI on that branch builds the report artifact on push.

        ## Contact
        If you need help reproducing the report or regenerating the VEP / DeepSEA comparisons, open an issue on the repository or email angel.morenu@ufl.edu.

        ---
        Last updated: 2025-11-15
