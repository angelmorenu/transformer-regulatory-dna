from pathlib import Path
import pandas as pd
import pytest


def find_metrics_path():
    p = Path('results') / 'metrics.csv'
    if p.exists():
        return p
    p2 = Path('notebooks') / 'results' / 'metrics.csv'
    if p2.exists():
        return p2
    pytest.skip('metrics.csv not found in expected locations')


def test_metrics_parsable_and_has_numeric():
    p = find_metrics_path()
    # read without strict schema
    df = pd.read_csv(p, header=None, dtype=str, on_bad_lines='skip')
    assert not df.empty, 'metrics.csv is empty'
    # try to detect at least one numeric between 0 and 1
    found = False
    for col in df.columns:
        for v in df[col].dropna().astype(str):
            try:
                f = float(v)
            except Exception:
                continue
            if 0.0 <= f <= 1.0:
                found = True
                break
        if found:
            break
    assert found, 'No numeric values in [0,1] found in metrics.csv (expected AUROC/PR-AUC)'
