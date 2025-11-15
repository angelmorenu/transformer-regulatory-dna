"""
Compute Spearman rank correlation and top-K enrichment between repo VEP outputs
and a DeepSEA benchmark file.

Usage:
    python src/compute_vep_vs_deepsea.py --deepsea notebooks/results/deepsea/deepsea_scores.tsv \
        --vep_tsv notebooks/results/vep/deltas_test.tsv \
        --topk_csv notebooks/results/plots/top50_test.csv \
        --out notebooks/results/plots/vep_deepsea_comparison.csv

The DeepSEA file should be a TSV/CSV with columns: record,pos,score
If the DeepSEA file uses genomic coords (chr,pos), the script will attempt to
match by (chr,pos) if your VEP outputs include 'record' encoding chromosome.

The script writes a small CSV with metrics and prints a concise summary.
"""
import argparse
import os
import sys
import csv
from collections import namedtuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import hypergeom


def read_vep_tsv(path):
    # vep tsv: columns include 'record', 'pos', 'max_abs' or dA,dC,dG,dT
    df = pd.read_csv(path, sep="\t", comment='#')
    # normalize column names
    cols = df.columns.str.strip()
    df.columns = cols
    if 'max_abs' in df.columns:
        df['vep_score'] = df['max_abs']
    else:
        # compute from allele deltas if present
        allele_cols = [c for c in ['dA','dC','dG','dT'] if c in df.columns]
        if allele_cols:
            df['vep_score'] = df[allele_cols].abs().max(axis=1)
        else:
            # fallback: if there's a numeric column with 'delta' in name
            delta_cols = [c for c in df.columns if 'delta' in c.lower() or 'd_' in c.lower()]
            if delta_cols:
                df['vep_score'] = df[delta_cols].abs().max(axis=1)
            else:
                raise ValueError('Could not find vep score columns in {}'.format(path))
    # ensure record and pos exist
    if 'record' not in df.columns or 'pos' not in df.columns:
        raise ValueError('VE P TSV must contain record and pos columns')
    return df[['record','pos','vep_score']].copy()


def read_deepsea(path):
    # Accept csv or tsv with columns 'record','pos','score' or 'chr','pos','score'
    ext = os.path.splitext(path)[1].lower()
    sep = ',' if ext == '.csv' else '\t'
    df = pd.read_csv(path, sep=sep, comment='#')
    cols = df.columns.str.strip()
    df.columns = cols
    if 'score' in df.columns:
        score_col = 'score'
    else:
        # try common variants
        cand = [c for c in df.columns if 'deep' in c.lower() or 'score' in c.lower() or 'delta' in c.lower()]
        if cand:
            score_col = cand[0]
        else:
            # fallback to last numeric column
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric:
                raise ValueError('No score column found in DeepSEA file')
            score_col = numeric[-1]
    df = df.rename(columns={score_col: 'deepsea_score'})
    if 'record' not in df.columns and 'chr' in df.columns:
        df = df.rename(columns={'chr':'record'})
    if 'record' not in df.columns or 'pos' not in df.columns:
        raise ValueError('DeepSEA file must contain record/chr and pos columns (or a compatible format)')
    return df[['record','pos','deepsea_score']].copy()


def align_and_merge(vep_df, deepsea_df):
    # merge on record,pos
    merged = pd.merge(vep_df, deepsea_df, on=['record','pos'], how='inner')
    return merged


def compute_spearman(merged):
    if merged.shape[0] < 3:
        return np.nan, np.nan
    rho, p = spearmanr(merged['vep_score'], merged['deepsea_score'])
    return float(rho), float(p)


def compute_topk_enrichment(vep_topk_csv, deepsea_df, K=50):
    # read our topk
    topk = pd.read_csv(vep_topk_csv)
    # expect columns rank,record,pos
    if 'record' not in topk.columns:
        topk_cols = [c for c in topk.columns if 'record' in c.lower()]
        if topk_cols:
            topk = topk.rename(columns={topk_cols[0]:'record'})
    if 'pos' not in topk.columns:
        pos_cols = [c for c in topk.columns if 'pos' in c.lower()]
        if pos_cols:
            topk = topk.rename(columns={pos_cols[0]:'pos'})
    topk = topk[['record','pos']].copy()
    # compute deepsea top-K by absolute score ranking
    deepsea_df = deepsea_df.copy()
    deepsea_df['abs_ds'] = deepsea_df['deepsea_score'].abs()
    deepsea_ranked = deepsea_df.sort_values('abs_ds', ascending=False)
    deepsea_topk = set(
        (r, int(p)) for r,p in zip(deepsea_ranked['record'].values[:K], deepsea_ranked['pos'].values[:K])
    )
    vep_topk = set((r, int(p)) for r,p in zip(topk['record'].values[:K], topk['pos'].values[:K]))
    overlap = len(vep_topk & deepsea_topk)
    # hypergeometric test: population = number of matched positions (intersection universe)
    # better to set population = total unique positions in merged deepsea and vep universe
    universe = len(pd.merge(vep_topk_to_df(vep_topk), deepsea_df, on=['record','pos'], how='outer').drop_duplicates())
    # If universe is zero fallback to len(deepsea_df)
    if universe == 0:
        universe = len(deepsea_df)
    # success in population = K (deepsea topk size)
    M = universe
    n = K
    N = K
    # hypergeom sf for >= overlap: sf(k-1)
    pval = hypergeom.sf(overlap-1, M, n, N)
    expected = (K * K) / float(M) if M>0 else np.nan
    fold = (overlap / expected) if expected and expected>0 else np.nan
    return {'K':K,'overlap':overlap,'universe':M,'pval':float(pval),'expected':expected,'fold':float(fold)}


def vep_topk_to_df(vep_topk_set):
    return pd.DataFrame([{'record':r,'pos':p} for r,p in vep_topk_set])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--deepsea', required=True, help='Path to DeepSEA scores (TSV/CSV)')
    p.add_argument('--vep_tsv', default='notebooks/results/vep/deltas_test.tsv', help='Path to VEP per-position TSV')
    p.add_argument('--topk_csv', default='notebooks/results/plots/top50_test.csv', help='Path to our top-K CSV')
    p.add_argument('--out', default='notebooks/results/plots/vep_deepsea_comparison.csv', help='CSV output path')
    p.add_argument('--Ks', default='10,25,50', help='Comma-separated K values to compute enrichment for')
    args = p.parse_args()

    vep_df = read_vep_tsv(args.vep_tsv)
    deepsea_df = read_deepsea(args.deepsea)
    merged = align_and_merge(vep_df, deepsea_df)
    rho, pval = compute_spearman(merged)
    print('Merged positions for comparison:', merged.shape[0])
    print('Spearman rho: {:.4g}, p-value: {:.4g}'.format(rho, pval))

    Ks = [int(x) for x in args.Ks.split(',') if x.strip()]
    enrichment_results = []
    for K in Ks:
        res = compute_topk_enrichment(args.topk_csv, deepsea_df, K=K)
        enrichment_results.append(res)
        print('K={K}: overlap={overlap}, fold={fold:.2g}, p={pval:.2g}'.format(**res))

    # write results to CSV
    rows = []
    rows.append({'metric':'spearman_rho','value':rho})
    rows.append({'metric':'spearman_pval','value':pval})
    for r in enrichment_results:
        rows.append({'metric':f"top{r['K']}_overlap", 'value':r['overlap']})
        rows.append({'metric':f"top{r['K']}_fold", 'value':r['fold']})
        rows.append({'metric':f"top{r['K']}_pval", 'value':r['pval']})

    outdf = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    outdf.to_csv(args.out, index=False)
    print('Wrote comparison CSV to', args.out)

if __name__ == '__main__':
    main()
