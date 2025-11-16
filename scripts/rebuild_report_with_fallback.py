#!/usr/bin/env python3
"""
Driver to regenerate report assets using the processed-test fallback and build the PDF.

Steps:
 - run scripts/write_vep_deepsea_tex.py to (re)generate macros
 - attempt to run pdflatex twice to build Morenu_CAP5510_ProjectReport.pdf
 - if PDF produced, add & commit it to git with a clear message

This script is conservative: if pdflatex is missing it prints instructions instead of failing.
"""
import subprocess
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / 'Morenu_CAP5510_ProjectReport.tex'
OUT_PDF = ROOT / 'Morenu_CAP5510_ProjectReport.pdf'
WRITE_SCRIPT = ROOT / 'scripts' / 'write_vep_deepsea_tex.py'


def run_write():
    print('Running write_vep_deepsea_tex.py...')
    res = subprocess.run([sys.executable, str(WRITE_SCRIPT)], cwd=str(ROOT))
    if res.returncode != 0:
        raise SystemExit('write_vep_deepsea_tex.py failed')


def build_pdf():
    pdflatex = shutil.which('pdflatex')
    if pdflatex is None:
        print('pdflatex not found on PATH. To build the PDF, install a TeX distribution (TeX Live / MacTeX) and run:')
        print(f'  pdflatex {TEX.name}\n  pdflatex {TEX.name}')
        return False
    # run twice for references
    cmd = [pdflatex, '-interaction=nonstopmode', TEX.name]
    print('Running:', ' '.join(cmd))
    for i in range(2):
        res = subprocess.run(cmd, cwd=str(ROOT))
        if res.returncode != 0:
            print('pdflatex failed (exit', res.returncode, '). See .log for details.')
            return False
    return True


def git_commit_pdf():
    if not OUT_PDF.exists():
        print('PDF not found, skipping git commit.')
        return
    # add and commit
    subprocess.run(['git', 'add', str(OUT_PDF)], cwd=str(ROOT))
    msg = 'Build report PDF (processed-test fallback): update vep/DeepSEA summary'
    subprocess.run(['git', 'commit', '-m', msg], cwd=str(ROOT))
    print('Committed PDF to git. Consider pushing branch to remote.')


def main():
    run_write()
    built = build_pdf()
    if built:
        print('PDF build succeeded')
        git_commit_pdf()
    else:
        print('PDF build skipped or failed. The macros were regenerated at:\n', ROOT / 'notebooks' / 'results' / 'plots' / 'vep_deepsea_summary.tex')


if __name__ == '__main__':
    main()
