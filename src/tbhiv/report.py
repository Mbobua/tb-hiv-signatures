"""Collects headline numbers, tables and figure paths and writes results/summary.md + results.json."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def md_table(df: pd.DataFrame, floatfmt="{:.3g}", index=True, max_rows=30) -> str:
    """Minimal markdown table (no 'tabulate' dependency)."""
    d = df.head(max_rows).copy()
    if index:
        d = d.reset_index()
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        if isinstance(v, (list, tuple)):
            return ", ".join(map(str, v))
        return str(v)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


class Report:
    def __init__(self, results_dir, title="Results summary"):
        self.dir = Path(results_dir)
        self.title = title
        self.sections: list[tuple[str, str]] = []
        self.numbers: dict = {}

    def add(self, heading: str, text: str = "", table: pd.DataFrame | None = None, figure=None, **tbl_kw):
        parts = [text] if text else []
        if table is not None:
            parts.append(md_table(table, **tbl_kw))
        if figure is not None:
            rel = os.path.relpath(Path(figure).resolve(), self.dir.resolve())
            parts.append(f"![{heading}]({Path(rel).as_posix()})")
        self.sections.append((heading, "\n\n".join(parts)))

    def number(self, key: str, value):
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        self.numbers[key] = value

    def write(self, data_note: str = "") -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        lines = [f"# {self.title}", "", f"_Generated {datetime.now():%Y-%m-%d %H:%M}_", ""]
        if data_note:
            lines += [f"> {data_note}", ""]
        for h, body in self.sections:
            lines += [f"## {h}", "", body, ""]
        out = self.dir / "summary.md"
        out.write_text("\n".join(lines))
        (self.dir / "results.json").write_text(json.dumps(self.numbers, indent=2, default=str))
        return out
