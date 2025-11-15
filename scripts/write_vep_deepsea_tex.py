#!/usr/bin/env python3
"""
Create a small LaTeX snippet with macros summarizing VEP vs DeepSEA metrics.

Behavior:
- If `notebooks/results/plots/vep_deepsea_comparison.csv` exists, read Spearman and top-K rows and
  write macros like \newcommand{\VepDeepSeaSpearman}{0.12} and top-K macros.
- Else if `notebooks/results/deepsea/processed_deepsea_predictions_per_sample.tsv` exists,
  write a fallback macro indicating the processed-test fallback and the number of samples.
- Else write macros indicating missing data.

The output file is `notebooks/results/plots/vep_deepsea_summary.tex` (created directories if needed).
"""
from pathlib import Path
import pandas as pd
import sys


ROOT = Path('.')
OUT_DIR = ROOT / 'notebooks' / 'results' / 'plots'
OUT_FILE = OUT_DIR / 'vep_deepsea_summary.tex'
CSV = ROOT / 'notebooks' / 'results' / 'plots' / 'vep_deepsea_comparison.csv'
FALLBACK = ROOT / 'notebooks' / 'results' / 'deepsea' / 'processed_deepsea_predictions_per_sample.tsv'


def write_missing(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as fh:
        fh.write('% Auto-generated summary: no DeepSEA comparison CSV found.\n')
        fh.write('\\newcommand{\\VepDeepSeaAvailable}{0}\\n')
        fh.write('% Provide notebooks/results/plots/vep_deepsea_comparison.csv to populate metrics.\n')
    print('WROTE', out_path)


def write_from_csv(csv_path: Path, out_path: Path):
    df = pd.read_csv(csv_path)
    # support a 2-column 'metric,value' format produced by the comparison script
    if set(df.columns) >= {"metric", "value"}:
        dmap = {r['metric']: r['value'] for _, r in df.to_dict(orient='index').items()}
    else:
        # fallback: create mapping from column names
        dmap = {c: df[c].iloc[0] if c in df.columns else '' for c in df.columns}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as fh:
        fh.write('% Auto-generated VEP vs DeepSEA summary macros\n')
        fh.write('\\newcommand{\\VepDeepSeaAvailable}{1}\\n')
        # spearman row if present
        # spearman
        rho = dmap.get('spearman_rho', '')
        pval = dmap.get('spearman_pval', '')
        fh.write(f'\\newcommand{{\\VepDeepSeaSpearman}}{{{rho}}}\\n')
        fh.write(f'\\newcommand{{\\VepDeepSeaSpearmanP}}{{{pval}}}\\n')

        # topK (use metric keys like top10_overlap, top10_fold, top10_pval)
        for K in [10,25,50]:
            pref = f'top{K}_'
            overlap = dmap.get(pref + 'overlap', '')
            fold = dmap.get(pref + 'fold', '')
            p = dmap.get(pref + 'pval', '')
            fh.write(f'\\newcommand{{\\VepDeepSeaTopK{K}Overlap}}{{{overlap}}}\\n')
            fh.write(f'\\newcommand{{\\VepDeepSeaTopK{K}Fold}}{{{fold}}}\\n')
            fh.write(f'\\newcommand{{\\VepDeepSeaTopK{K}P}}{{{p}}}\\n')

    print('WROTE', out_path)


def write_fallback(fallback_path: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(fallback_path, sep='\t')
    nsamples = df.shape[0]
    max_score = df['max_score'].astype(float).max()
    with open(out_path, 'w') as fh:
        fh.write('% Auto-generated VEP vs DeepSEA summary (processed-test fallback)\n')
        fh.write('\\newcommand{\\VepDeepSeaAvailable}{2}\\n')
        fh.write(f'\\newcommand{{\\VepDeepSeaFallbackSamples}}{{{nsamples}}}\\n')
        fh.write(f'\\newcommand{{\\VepDeepSeaFallbackMaxScore}}{{{max_score:.4g}}}\\n')
    print('WROTE fallback', out_path)


def main():
    if CSV.exists():
        write_from_csv(CSV, OUT_FILE)
        return
    if FALLBACK.exists():
        write_fallback(FALLBACK, OUT_FILE)
        return
    write_missing(OUT_FILE)


if __name__ == '__main__':
    main()
