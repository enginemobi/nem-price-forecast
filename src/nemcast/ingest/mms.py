"""Parser for AEMO MMS "CSV" report files (I/D/C row format), including zips of zips."""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

TableKey = tuple[str, str]


def iter_csv_texts(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield `(member_name, csv_text)` for every CSV in a zip, descending into nested zips."""
    with zipfile.ZipFile(path) as zf:
        yield from _iter_zip(zf)


def _iter_zip(zf: zipfile.ZipFile) -> Iterator[tuple[str, str]]:
    for name in zf.namelist():
        lower = name.lower()
        if lower.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(zf.read(name))) as inner:
                for inner_name, text in _iter_zip(inner):
                    yield f"{name}/{inner_name}", text
        elif lower.endswith(".csv"):
            yield name, zf.read(name).decode("utf-8", errors="replace")


def parse_mms_text(
    text: str,
    tables: dict[TableKey, list[str]],
) -> dict[TableKey, list[list[str]]]:
    """Extract rows of the requested `(report, table)` keys, keeping only the listed columns."""
    out: dict[TableKey, list[list[str]]] = {k: [] for k in tables}
    index: dict[TableKey, list[int] | None] = {}
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        kind = row[0]
        if kind == "I":
            key = (row[1], row[2])
            if key in tables:
                header = row[4:]
                index[key] = [header.index(c) for c in tables[key]]
            else:
                index[key] = None
        elif kind == "D":
            key = (row[1], row[2])
            idx = index.get(key)
            if idx is not None:
                values = row[4:]
                out[key].append([values[i] if i < len(values) else "" for i in idx])
    return out


def read_tables(
    paths: Iterable[str | Path],
    tables: dict[TableKey, list[str]],
    with_member: bool = False,
) -> dict[TableKey, pd.DataFrame]:
    """Read the requested tables from MMS zip files into DataFrames (string-typed).

    With `with_member`, a `SOURCE_FILE` column records the inner CSV name each row came from.
    """
    rows: dict[TableKey, list[list[str]]] = {k: [] for k in tables}
    for path in paths:
        for member, text in iter_csv_texts(path):
            for key, found in parse_mms_text(text, tables).items():
                if with_member:
                    found = [r + [member] for r in found]
                rows[key].extend(found)
    frames = {}
    for key, cols in tables.items():
        columns = cols + (["SOURCE_FILE"] if with_member else [])
        frames[key] = pd.DataFrame(rows[key], columns=columns)
    return frames
