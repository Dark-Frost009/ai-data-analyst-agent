"""
Tests for app.core.data_loader.load_csv.

Uses a small BytesIO-based fake to stand in for Streamlit's UploadedFile
(which exposes .read(), .seek(), .name, and .size) so these tests run
without any Streamlit runtime or network access.
"""

import io

import pytest

from app.core.data_loader import (
    CSVEncodingError,
    CSVParsingError,
    EmptyFileError,
    FileTooLargeError,
    load_csv,
)


class FakeUploadedFile(io.BytesIO):
    """Mimics the parts of Streamlit's UploadedFile interface this module uses."""

    def __init__(self, data: bytes, name: str = "test.csv"):
        super().__init__(data)
        self.name = name
        self.size = len(data)


# --------------------------------------------------------------------------
# Success cases
# --------------------------------------------------------------------------
def test_load_csv_success_basic():
    content = b"name,age,city\nAlice,30,NYC\nBob,25,LA\n"
    df = load_csv(FakeUploadedFile(content))

    assert list(df.columns) == ["name", "age", "city"]
    assert len(df) == 2
    assert df.iloc[0]["name"] == "Alice"
    assert int(df.iloc[1]["age"]) == 25


def test_load_csv_strips_utf8_bom():
    content = "name,age\nAlice,30\n".encode("utf-8-sig")
    df = load_csv(FakeUploadedFile(content))

    # The BOM should not leak into the first column name.
    assert list(df.columns) == ["name", "age"]


def test_load_csv_falls_back_to_cp1252():
    # 0x96 is an en-dash in cp1252 but is not valid standalone UTF-8.
    content = "name,note\nAlice,pre\u20131950\n".encode("cp1252")
    df = load_csv(FakeUploadedFile(content))

    assert list(df.columns) == ["name", "note"]
    assert "1950" in df.iloc[0]["note"]


def test_load_csv_header_only_returns_empty_dataframe():
    content = b"name,age,city\n"
    df = load_csv(FakeUploadedFile(content))

    assert list(df.columns) == ["name", "age", "city"]
    assert len(df) == 0


def test_load_csv_accepts_plain_bytesio_without_name_or_size():
    # A generic file-like object, not a Streamlit UploadedFile subclass —
    # exercises the fallback path where .size isn't available.
    content = b"a,b\n1,2\n"
    df = load_csv(io.BytesIO(content))

    assert list(df.columns) == ["a", "b"]
    assert len(df) == 1


def test_load_csv_respects_max_size_override_when_within_limit():
    content = b"a,b\n1,2\n"
    df = load_csv(FakeUploadedFile(content), max_size_mb=1)
    assert len(df) == 1


# --------------------------------------------------------------------------
# Failure cases
# --------------------------------------------------------------------------
def test_load_csv_rejects_none():
    with pytest.raises(ValueError):
        load_csv(None)


def test_load_csv_rejects_zero_byte_file():
    with pytest.raises(EmptyFileError):
        load_csv(FakeUploadedFile(b""))


def test_load_csv_rejects_whitespace_only_content():
    with pytest.raises(EmptyFileError):
        load_csv(FakeUploadedFile(b"\n\n\n"))


def test_load_csv_rejects_malformed_csv():
    # Header declares 2 columns; the second data row has 3 fields.
    content = b"a,b\n1,2\n3,4,5\n"
    with pytest.raises(CSVParsingError):
        load_csv(FakeUploadedFile(content))


def test_load_csv_rejects_undecodable_bytes():
    # 0x81 is invalid as standalone UTF-8 and undefined in cp1252 —
    # fails both fallback encodings.
    content = b"a,b\n\x81,2\n"
    with pytest.raises(CSVEncodingError):
        load_csv(FakeUploadedFile(content))


def test_load_csv_rejects_oversized_file_using_declared_size():
    # 2 KB of content with a 0.001 MB (~1 KB) limit — caught by the fast
    # pre-check via the .size attribute, before the file is even read.
    content = b"a,b\n" + b"1,2\n" * 300
    with pytest.raises(FileTooLargeError):
        load_csv(FakeUploadedFile(content), max_size_mb=0.001)


def test_load_csv_rejects_oversized_file_without_declared_size():
    # Plain BytesIO has no .size attribute, so this exercises the
    # defensive post-read size check instead of the fast pre-check.
    content = b"a,b\n" + b"1,2\n" * 300
    with pytest.raises(FileTooLargeError):
        load_csv(io.BytesIO(content), max_size_mb=0.001)