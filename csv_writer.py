# csv_writer.py
import csv
from pathlib import Path

class CsvWriter:
    def __init__(self, out_path, header):
        self.out_path = Path(out_path)
        self.header = list(header)
        self._file = None
        self._writer = None

    def open(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.out_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.header)
        self._writer.writeheader()

    def write_row(self, row: dict):
        # Fill missing keys with empty value so LK/ORB can differ safely
        safe_row = {k: row.get(k, "") for k in self.header}
        self._writer.writerow(safe_row)

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None
