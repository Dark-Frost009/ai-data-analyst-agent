"""Bounded CSV preparation and session-private, disk-backed execution.

Only trusted preparation SQL writes the database. Generated SQL uses the
existing executor's validation, timeout, memory and external-access protections.
"""
import csv
import shutil
import tempfile
import time
import weakref
import threading
import sys
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from app.config import config
from app.utils.temporary_storage import runtime_directory
from app.core.data_loader import CSVParsingError, EmptyFileError, FileTooLargeError, _normalize_column_names
from app.core.data_profiler import profile_dataframe
from app.core.execution_preparation import _coerce_datetime
from app.core.numeric_preparation import coerce_numeric
from app.core.semantic_checks import check_conversion_fidelity
from app.core.sql_executor import SQLExecutor, QueryExecutionError
from app.utils.dataset_resources import DatasetMemoryBudget, DatasetResourceError

disk_budget = DatasetMemoryBudget(config.max_total_dataset_disk_mb * 1024**2)


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


class DiskDataset:
    """Own raw CSV and typed database until cleared or garbage collected."""

    def __init__(self, upload, progress=None):
        try:
            reservation = disk_budget.reserve(config.max_dataset_disk_mb * 1024**2)
        except DatasetResourceError:
            raise DatasetResourceError("Temporary dataset storage is full. Clear an unused dataset and retry.") from None
        try:
            self.root = Path(tempfile.mkdtemp(prefix="dataset-", dir=runtime_directory()))
        except Exception:
            reservation.release()
            raise
        self._finalizer = weakref.finalize(self, self._cleanup, self.root, reservation)
        self.path = self.root / "execution.duckdb"
        self._users = 0
        self._retired = False
        self._lock = threading.Lock()
        self.raw = self.root / "source.csv"
        self.started = time.monotonic()
        try:
            upload.seek(0)
            size = 0
            with self.raw.open("wb") as target:
                while block := upload.read(1024**2):
                    size += len(block)
                    if size > config.max_upload_size_mb * 1024**2:
                        raise FileTooLargeError("CSV exceeds the configured upload limit.")
                    target.write(block)
                    self._check_budget()
            if not size:
                raise EmptyFileError("The uploaded CSV is empty.")
            self.encoding = "utf-8-sig"
            try:
                with self.raw.open(encoding=self.encoding) as source:
                    while source.read(1024**2):
                        self._check_budget()
            except UnicodeDecodeError:
                self.encoding = "cp1252"
            self.columns = None
            self.kinds = None
            self.us_evidence = None
            self.row_count = 0
            self.nulls = None
            self.non_nulls = None
            samples = []
            for batch in self.batches():
                if self.kinds is None:
                    self.kinds = [{"iso", "us", "Int64", "Float64"} for _ in self.columns]
                    self.us_evidence = [False] * len(self.columns)
                    self.nulls = [0] * len(self.columns)
                    self.non_nulls = [0] * len(self.columns)
                self.row_count += len(batch)
                if self.row_count > config.max_disk_dataset_rows:
                    raise DatasetResourceError("CSV exceeds the disk dataset row limit.")
                if not samples:
                    # One bounded batch, including its backing arrays, is retained.
                    samples.append(batch.head(1000).copy(deep=True))
                for i, name in enumerate(self.columns):
                    values = batch[name].dropna()
                    self.nulls[i] += int(batch[name].isna().sum())
                    self.non_nulls[i] += len(values)
                    if values.empty:
                        continue
                    iso = values.str.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}").all()
                    us = values.str.fullmatch(r"[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}").all()
                    if not iso or _coerce_datetime(batch[name]) is None:
                        self.kinds[i].discard("iso")
                    if us:
                        parts = [v.split("/") for v in values]
                        self.us_evidence[i] |= any(int(p[1]) > 12 for p in parts)
                        # Sentinel supplies ordering evidence only, never stored.
                        check = pd.concat([batch[name], pd.Series(["12/31/2000"])], ignore_index=True)
                        us = _coerce_datetime(check) is not None
                    if not us:
                        self.kinds[i].discard("us")
                    numeric = coerce_numeric(batch[name])
                    for kind in ("Int64", "Float64"):
                        if numeric is None or str(numeric.dtype) != kind:
                            self.kinds[i].discard(kind)
                if progress:
                    progress(f"Checked {self.row_count:,} rows")
                self._check_budget()
            if not self.row_count:
                raise EmptyFileError("CSV contains no data rows.")
            self.types = [next((k for k in ("iso", "us", "Int64", "Float64")
                               if self.non_nulls[i] and k in kinds and (k != "us" or self.us_evidence[i])), "text")
                          for i, kinds in enumerate(self.kinds)]
            self.preview = pd.concat(samples, ignore_index=True)
            self.raw_profile = self._profile(self.preview)
            self.execution_profile = self._profile(self.convert(self.preview))
            conn = duckdb.connect(str(self.path))
            try:
                conn.execute(f"SET memory_limit='{config.duckdb_memory_limit}'")
                conn.execute(f"SET threads={config.duckdb_thread_limit}")
                conn.execute("SET temp_directory=''")
                conn.execute("SET enable_external_access=false")
                sql_types = {"iso": "TIMESTAMP", "us": "TIMESTAMP", "Int64": "BIGINT", "Float64": "DOUBLE", "text": "VARCHAR"}
                definitions = ', '.join(f"{_quote(n)} {sql_types[t]}" for n, t in zip(self.columns, self.types))
                conn.execute(f"CREATE TABLE dataset ({definitions})")
                inserted = 0
                for batch in self.batches():
                    conn.register("prepared_batch", self.convert(batch))
                    conn.execute("INSERT INTO dataset SELECT * FROM prepared_batch")
                    conn.unregister("prepared_batch")
                    inserted += len(batch)
                    conn.execute("CHECKPOINT")
                    self._check_budget()
                    if progress:
                        progress(f"Prepared {inserted:,} of {self.row_count:,} rows")
            finally:
                conn.close()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _cleanup(root, reservation):
        try:
            shutil.rmtree(root)
        except FileNotFoundError:
            pass
        finally:
            reservation.release()

    def close(self):
        with self._lock:
            self._retired = True
            if not self._users:
                self._finalizer()

    def acquire(self):
        with self._lock:
            if self._retired:
                raise DatasetResourceError("Dataset has been cleared. Upload it again.")
            self._users += 1

    def release(self):
        with self._lock:
            self._users -= 1
            if self._retired and not self._users:
                self._finalizer()

    def _check_budget(self):
        if time.monotonic() - self.started > config.dataset_load_timeout_seconds:
            raise DatasetResourceError("CSV preparation timed out. Upload a smaller subset.")
        if sum(p.stat().st_size for p in self.root.iterdir() if p.is_file()) > config.max_dataset_disk_mb * 1024**2:
            raise DatasetResourceError("Dataset exceeds the temporary disk budget.")
        if shutil.disk_usage(self.root).free < 64 * 1024**2:
            raise DatasetResourceError("Temporary disk space is low. Try a smaller CSV.")

    def batches(self):
        """Strict CSV parsing with both row and byte bounds on each batch."""
        csv.field_size_limit(1024**2)
        try:
            with self.raw.open(encoding=self.encoding, newline="") as source:
                def bounded_lines():
                    while line := source.readline(2 * 1024**2 + 1):
                        if len(line) > 2 * 1024**2:
                            raise CSVParsingError("CSV contains an oversized physical line.")
                        yield line
                reader = csv.reader(bounded_lines(), strict=True)
                header = next((r for r in reader if r), None)
                if not header:
                    raise EmptyFileError("CSV has no header.")
                if len(header) > config.max_dataset_columns:
                    raise DatasetResourceError("CSV has too many columns.")
                names = list(_normalize_column_names(pd.DataFrame(columns=header)).columns)
                self.columns = names
                rows, size = [], 0
                for row in reader:
                    if not row:
                        continue
                    if len(row) != len(names):
                        raise CSVParsingError("CSV rows must have the same number of fields as the header.")
                    row_size = sum(len(v.encode("utf-8")) + 64 for v in row)
                    if row_size > 4 * 1024**2:
                        raise CSVParsingError("CSV contains an oversized row.")
                    rows.append([v if v != "" else None for v in row])
                    size += row_size
                    if len(rows) >= 5000 or size >= 4 * 1024**2:
                        yield pd.DataFrame(rows, columns=names, dtype=object)
                        rows, size = [], 0
                if rows:
                    yield pd.DataFrame(rows, columns=names, dtype=object)
        except (csv.Error, UnicodeError) as exc:
            raise CSVParsingError("CSV contains invalid encoding, quoting, or an oversized field.") from exc

    def convert(self, batch):
        result = batch.copy(deep=True)
        for name, kind in zip(self.columns, self.types):
            if kind in ("iso", "us"):
                result[name] = pd.to_datetime(batch[name], format="%Y-%m-%d" if kind == "iso" else "%m/%d/%Y", errors="raise")
            elif kind in ("Int64", "Float64"):
                result[name] = pd.array([None if pd.isna(v) else int(v) if kind == "Int64" else float(v) for v in batch[name]], dtype=kind)
        return result

    def _profile(self, sample):
        profile = profile_dataframe(sample)
        profile.row_count = self.row_count
        profile.statistics_sampled = self.row_count > len(sample)
        profile.profiled_row_count = len(sample)
        for i, column in enumerate(profile.columns):
            column.null_count = self.nulls[i]
            column.null_percentage = round(self.nulls[i] / self.row_count * 100, 2)
        return profile

    def check_conversions(self, sql):
        # The full dataset is checked in bounded batches, never only the preview.
        import sqlglot
        from sqlglot import exp
        if not any(cast.find(exp.Column) is not None for cast in sqlglot.parse_one(sql, read="duckdb").find_all(exp.Cast)):
            check_conversion_fidelity(sql, self.convert(self.preview))
            return
        started = time.monotonic()
        for batch in self.batches():
            check_conversion_fidelity(sql, self.convert(batch))
            if time.monotonic() - started > config.query_timeout_seconds:
                raise DatasetResourceError("Conversion checks timed out. Query the prepared column types directly.")


class DiskSQLExecutor(SQLExecutor):
    """Reuse SQLExecutor's execution protections with a read-only database."""

    def __init__(self, dataset, max_result_rows=None):
        self._dataset = dataset  # Retain files until timeout workers finish.
        self._table_name = "dataset"
        self._max_result_rows = config.max_query_result_rows if max_result_rows is None else max_result_rows
        if self._max_result_rows <= 0:
            raise ValueError("max_result_rows must be greater than zero")
        self._timeout_seconds = config.query_timeout_seconds
        self._timeout_cleanup_pending = False
        self._released = False
        dataset.acquire()
        try:
            self._conn = duckdb.connect(str(dataset.path), read_only=True)
            self._conn.execute(f"SET memory_limit='{config.duckdb_memory_limit}'")
            self._conn.execute(f"SET threads={config.duckdb_thread_limit}")
            self._conn.execute("SET temp_directory=''")
            self._conn.execute("SET enable_external_access=false")
            self._conn = _BoundedConnection(self._conn, self._max_result_rows)
        except Exception:
            if hasattr(self, "_conn"):
                self._conn.close()
            dataset.release()
            raise

    def _release_dataset(self):
        if not self._released:
            self._released = True
            self._dataset.release()

    def close(self):
        if not self._timeout_cleanup_pending and not self._released:
            try:
                super().close()
            finally:
                self._release_dataset()

    def _close_after_worker_finishes(self, future):
        try:
            super()._close_after_worker_finishes(future)
        finally:
            self._release_dataset()


class _BoundedConnection:
    """Keep a wide query result from expanding into an unbounded DataFrame."""

    def __init__(self, connection, row_limit):
        self.connection = connection
        self.row_limit = row_limit

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def execute(self, sql):
        return _BoundedCursor(self.connection.execute(sql), self.row_limit)


class _BoundedCursor:
    def __init__(self, cursor, row_limit):
        self.cursor = cursor
        self.row_limit = row_limit

    def __getattr__(self, name):
        return getattr(self.cursor, name)

    def fetchdf(self):
        rows, size = [], 0
        names = [column[0] for column in self.cursor.description]
        while row := self.cursor.fetchone():
            size += sys.getsizeof(row) + sum(sys.getsizeof(v) for v in row)
            if size > 16 * 1024**2 or len(rows) >= self.row_limit:
                raise QueryExecutionError("Result is too large to display. Select fewer columns or aggregate the data.")
            rows.append(row)
        # Explicit SQL types avoid pandas converting nullable large integers
        # through float64 (which rounds values beyond 2**53).
        frame = pd.DataFrame(rows, columns=names, dtype=object)
        signed = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT"}
        unsigned = {"UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT"}
        for i, (_, sql_type, *_) in enumerate(self.cursor.description):
            kind = str(sql_type)
            values = [row[i] for row in rows]
            present = [v for v in values if v is not None]
            # Older DuckDB versions expose NUMBER/STRING/DATETIME rather
            # than concrete SQL types. Inspect Python scalars without a
            # float intermediary, and keep unrepresentable values as objects.
            if kind == "NUMBER" and present:
                if all(type(v) is int and -(2**63) <= v < 2**63 for v in present):
                    kind = "BIGINT"
                elif all(type(v) is float for v in present):
                    kind = "DOUBLE"
            elif kind == "DATETIME" and present and all(isinstance(v, (date, datetime)) and getattr(v, "tzinfo", None) is None for v in present):
                kind = "TIMESTAMP"
            if kind in signed | unsigned:
                frame.isetitem(i, pd.array(values, dtype="Int64" if kind in signed else "UInt64"))
            elif kind == "HUGEINT" and all(v is None or -(2**63) <= v < 2**63 for v in values):
                frame.isetitem(i, pd.array(values, dtype="Int64"))
            elif kind in {"FLOAT", "DOUBLE"}:
                frame.isetitem(i, pd.array(values, dtype="Float64"))
            elif kind == "BOOLEAN":
                frame.isetitem(i, pd.array(values, dtype="boolean"))
            elif (kind.startswith("TIMESTAMP") and kind != "TIMESTAMP WITH TIME ZONE") or kind == "DATE":
                frame.isetitem(i, pd.to_datetime(values))
        return frame
