import io
import zipfile

from nemcast.ingest import mms
from nemcast.ingest.archive import file_dates, select_files

SAMPLE = """C,NEMP.WORLD,DISPATCHIS,AEMO,PUBLIC,2026/11/06,00:00:33
I,DISPATCH,PRICE,5,SETTLEMENTDATE,RUNNO,REGIONID,INTERVENTION,RRP
D,DISPATCH,PRICE,5,"2026/11/06 00:05:00",1,NSW1,0,85.5
D,DISPATCH,PRICE,5,"2026/11/06 00:05:00",1,SA1,0,-12.0
I,DISPATCH,OTHER,1,A,B
D,DISPATCH,OTHER,1,1,2
C,"END OF REPORT",7
"""


def test_parse_mms_text_selects_table_and_columns():
    out = mms.parse_mms_text(SAMPLE, {("DISPATCH", "PRICE"): ["REGIONID", "RRP"]})
    assert out[("DISPATCH", "PRICE")] == [["NSW1", "85.5"], ["SA1", "-12.0"]]


def test_read_tables_handles_nested_zip(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("PUBLIC_DISPATCHIS_202611060005_20261106000033.CSV", SAMPLE)
    outer = tmp_path / "PUBLIC_DISPATCHIS_20261106.zip"
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("PUBLIC_DISPATCHIS_202611060005_20261106000033.zip", inner.getvalue())
    frames = mms.read_tables([outer], {("DISPATCH", "PRICE"): ["SETTLEMENTDATE", "RRP"]}, with_member=True)
    df = frames[("DISPATCH", "PRICE")]
    assert len(df) == 2
    assert df["SOURCE_FILE"].iloc[0].endswith(".CSV")


def test_file_dates_and_selection():
    daily = "PUBLIC_P5MIN_20261106.zip"
    weekly = "PUBLIC_PREDISPATCHIS_20261025_20261031.zip"
    assert str(file_dates(daily)[0]) == "2026-11-06"
    assert [str(d) for d in file_dates(weekly)] == ["2026-10-25", "2026-10-31"]
    from datetime import date

    assert select_files([daily, weekly], date(2026, 10, 30), date(2026, 11, 1)) == [weekly]
