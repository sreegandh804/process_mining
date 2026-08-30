#!/usr/bin/env python3
"""Regenerate invoices.xlsx / payments.xlsx from the committed CSVs.

The CSVs are the source of truth; the .xlsx files are committed too so the
`run_tabular.py --xlsx` (real Excel) path is turnkey. Run this only if you edit
the CSVs. Needs openpyxl (`pip install openpyxl`).
"""
import csv
import datetime
from pathlib import Path


def to_xlsx(csv_path: Path, xlsx_path: Path) -> None:
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    with csv_path.open(newline="") as f:
        for row in csv.reader(f):
            out = []
            for cell in row:
                c = cell.strip()
                try:
                    out.append(datetime.date.fromisoformat(c))  # store ISO dates as real dates
                    continue
                except ValueError:
                    pass
                out.append(c)
            ws.append(out)
    wb.save(xlsx_path)
    print("wrote", xlsx_path)


if __name__ == "__main__":
    base = Path(__file__).parent
    to_xlsx(base / "invoices.csv", base / "invoices.xlsx")
    to_xlsx(base / "payments.csv", base / "payments.xlsx")
