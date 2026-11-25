"""Run every stage end-to-end without the notebook.

    python scripts/run_pipeline.py                     # real data from GEO
    python scripts/run_pipeline.py --repeats 2         # quicker CV
    python scripts/run_pipeline.py --matrix GSE37250=path/to/GSE37250_series_matrix.txt.gz
    python scripts/run_pipeline.py --demo              # SYNTHETIC data, offline code check only
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbhiv import Config  # noqa: E402
from tbhiv.pipeline import run_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument("--repeats", type=int, default=5, help="CV repeats (default 5)")
    ap.add_argument("--matrix", action="append", default=[], metavar="GSE=PATH",
                    help="use a local series-matrix file instead of downloading")
    ap.add_argument("--offline", action="store_true", help="skip Enrichr/STRING (co-expression network instead)")
    ap.add_argument("--demo", action="store_true", help="SYNTHETIC data - tests the code, produces no results")
    a = ap.parse_args()

    cfg = Config(data_dir=a.data, results_dir=a.results, cv_repeats=a.repeats, online=not a.offline,
                 local_matrix_files=dict(m.split("=", 1) for m in a.matrix))
    note = ""
    if a.demo:
        from tbhiv.synthetic import make_synthetic_geo

        syn = make_synthetic_geo(Path(a.data) / "synthetic", seed=1)
        cfg.local_matrix_files, cfg.local_platform_files = syn["local_matrix_files"], syn["local_platform_files"]
        cfg.gmt_files, cfg.online = syn["gmt_files"], False
        cfg.results_dir = Path("results_SYNTHETIC_DEMO")
        note = "SYNTHETIC DEMO DATA - these numbers are NOT research results."
    out = run_all(cfg, data_note=note)
    print("\n" + out["conclusions"])
    print(f"\nSummary: {out['report_path']}")


if __name__ == "__main__":
    main()
