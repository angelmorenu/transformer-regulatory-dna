Place DeepSEA per-position (or per-variant) prediction scores here for VEP comparisons.

Expected filename (suggested):
  notebooks/results/deepsea/deepsea_scores.tsv

Expected format (TSV/CSV):
  record	pos	score

Examples of `record` formatting the script can match:
  - If your DeepSEA outputs used sample indices: idx:7
  - If genomic coordinates: chr1,chr2 etc. (then record should be like chr1)

Example row (TSV):
  idx:7	1999	4.6e-05
  chr1	123456	0.00012

Notes:
  - The script `src/compute_vep_vs_deepsea.py` merges on (record,pos). Ensure your record/pos match the VEP outputs (see `notebooks/results/vep/deltas_test.tsv` and `notebooks/results/plots/top50_test.csv`).
  - If DeepSEA predictions use different naming (e.g., chromosome and 1-based coordinates) you may need to pre-process to match the VEP naming convention.

After placing `deepsea_scores.tsv` run:

```bash
python src/compute_vep_vs_deepsea.py --deepsea notebooks/results/deepsea/deepsea_scores.tsv
```

The script will write `notebooks/results/plots/vep_deepsea_comparison.csv` with Spearman and top-K enrichment results.
