"""Pull recent history (about the last 13 months) from the NEMWeb `Reports/Archive` folders into Parquet."""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from nemcast.ingest import mms, nemweb

RAW_DIR = Path("data/raw")
LAKE_DIR = Path("data/lake")

DISPATCH_TABLES = {
    ("DISPATCH", "PRICE"): ["SETTLEMENTDATE", "REGIONID", "INTERVENTION", "RRP"],
    ("DISPATCH", "REGIONSUM"): [
        "SETTLEMENTDATE",
        "REGIONID",
        "INTERVENTION",
        "TOTALDEMAND",
        "AVAILABLEGENERATION",
        "NETINTERCHANGE",
    ],
}
PREDISPATCH_TABLES = {
    ("PREDISPATCH", "REGION_PRICES"): ["DATETIME", "REGIONID", "INTERVENTION", "RRP"],
    ("PREDISPATCH", "REGION_SOLUTION"): [
        "DATETIME",
        "REGIONID",
        "INTERVENTION",
        "TOTALDEMAND",
        "AVAILABLEGENERATION",
        "NETINTERCHANGE",
    ],
}
P5MIN_TABLES = {
    ("P5MIN", "REGIONSOLUTION"): [
        "RUN_DATETIME",
        "INTERVAL_DATETIME",
        "REGIONID",
        "INTERVENTION",
        "RRP",
        "TOTALDEMAND",
        "AVAILABLEGENERATION",
        "NETINTERCHANGE",
    ],
}

SOURCES = {
    "dispatch": ("Reports/Archive/DispatchIS_Reports", DISPATCH_TABLES),
    "predispatch": ("Reports/Archive/PredispatchIS_Reports", PREDISPATCH_TABLES),
    "p5min": ("Reports/Archive/P5_Reports", P5MIN_TABLES),
}
_TIME_COLS = ("SETTLEMENTDATE", "DATETIME", "RUN_DATETIME", "INTERVAL_DATETIME")
_NUM_COLS = ("RRP", "TOTALDEMAND", "AVAILABLEGENERATION", "NETINTERCHANGE")


def file_dates(name: str) -> tuple[date, date]:
    """Date range covered by an archive file (daily `_YYYYMMDD.zip` or weekly `_YYYYMMDD_YYYYMMDD.zip`)."""
    found = re.findall(r"_(\d{8})(?=[_.])", name)
    first = datetime.strptime(found[0], "%Y%m%d").date()
    last = datetime.strptime(found[-1], "%Y%m%d").date()
    return first, last


def select_files(names: list[str], start: date, end: date) -> list[str]:
    return [n for n in names if file_dates(n)[1] >= start and file_dates(n)[0] <= end]


def _run_times_from_member(member: str) -> tuple[str, str]:
    """`(first interval, publish time)` from e.g. `PUBLIC_PREDISPATCHIS_202610250030_20261025000133`."""
    m = re.search(r"_(\d{12})_(\d{14})", member)
    return m.group(1), m.group(2)


def parse_file(source: str, path: Path) -> dict[str, pd.DataFrame]:
    tables = SOURCES[source][1]
    frames = mms.read_tables([path], tables, with_member=source == "predispatch")
    out = {}
    for (_, table), df in frames.items():
        df = df[df["INTERVENTION"] == "0"].drop(columns="INTERVENTION")
        if "SOURCE_FILE" in df:
            stamps = df.pop("SOURCE_FILE").map(_run_times_from_member)
            df.insert(0, "RUN_DATETIME", pd.to_datetime(stamps.str[0], format="%Y%m%d%H%M"))
            df.insert(1, "PUBLISHED", pd.to_datetime(stamps.str[1], format="%Y%m%d%H%M%S"))
        for c in df.columns:
            if c in _TIME_COLS and pd.api.types.is_string_dtype(df[c]):
                df[c] = pd.to_datetime(df[c], format="%Y/%m/%d %H:%M:%S")
            elif c in _NUM_COLS:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        out[f"{source}_{table.lower()}"] = df.drop_duplicates()
    return out


def fetch(source: str, start: date, end: date, workers: int = 2) -> None:
    folder, _ = SOURCES[source]
    names = select_files(nemweb.list_files(folder), start, end)
    raw = RAW_DIR / source
    with ThreadPoolExecutor(4) as pool:
        paths = list(pool.map(lambda n: nemweb.download(folder, n, raw), names))
    with ProcessPoolExecutor(workers) as pool:
        for name, parsed in zip(names, pool.map(parse_file, [source] * len(paths), paths), strict=True):
            for table, df in parsed.items():
                dest = LAKE_DIR / table / f"{Path(name).stem}.parquet"
                dest.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(dest, index=False)
            print(f"{source}: {name} -> {sum(len(d) for d in parsed.values())} rows", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sources", nargs="+", choices=sorted(SOURCES))
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, default=date.today() - timedelta(days=1))
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    for s in args.sources:
        fetch(s, args.start, args.end, args.workers)


if __name__ == "__main__":
    main()
