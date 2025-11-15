import os
from pathlib import Path


def test_write_vep_deepsea_summary():
    """Run the summary writer and assert the LaTeX snippet is produced with the expected macro."""
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / 'scripts' / 'write_vep_deepsea_tex.py'
    assert script.exists(), f"Script missing: {script}"
    # import and run
    import importlib.util
    spec = importlib.util.spec_from_file_location('write_vep_deepsea_tex', str(script))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # run main
    if hasattr(mod, 'main'):
        mod.main()
    out = repo_root / 'notebooks' / 'results' / 'plots' / 'vep_deepsea_summary.tex'
    assert out.exists(), 'vep_deepsea_summary.tex not created'
    content = out.read_text()
    assert '\\newcommand{\\VepDeepSeaAvailable}' in content
