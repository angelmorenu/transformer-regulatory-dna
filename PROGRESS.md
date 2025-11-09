
---

### ✅ **Updated `PROGRESS.md`**
Extends your week tracker to include fine-tuning and variant-effect evaluation milestones:

```markdown
# Project Progress Tracker

---

### **Week 1 – Data Pipeline & Preprocessing** ✅ *Completed*
- Sequence window extraction and dataset serialization (`.npz`)
- BED/FASTA parsing utilities for reference genomes
- Split generation: train / validation / test

---

### **Week 2 – CNN Baselines** ✅ *Completed*
- Implemented Basset-style CNN
- AUROC / PR-AUC metrics and training harness
- Checkpointing and metrics logging (`results/metrics.csv`)

---

### **Week 3 – Transformer Fine-Tuning** ✅ *Completed*
- Linear probe over TinyEnc / DNABERT-2 embeddings  
- Gradient clipping + hyperparameter tuning
- Run configuration snapshot + per-epoch metrics CSV
- Validation curves and saved best checkpoint → `results/probe_best.pt`

---

### **Week 4 – Variant Effect Prediction** ✅ *Completed*
- In-silico saturation mutagenesis pipeline (`04_variant_effects.ipynb`)
- Support for BED/FASTA custom regions
- Exports: `deltas_<split>.npz`, `deltas_<split>.tsv`, `top50_<split>.tsv`
- Summary table and heatmap visualizations

---

### **Next: Week 5 – Results Integration & Reporting** 🚧 *In Progress*
- Aggregate metrics into comparative plots
- Compute variant-effect correlations vs DeepSEA
- Generate publication-ready figures & LaTeX report sections
