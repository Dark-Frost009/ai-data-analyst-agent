"""
CSV data loading.

Reads an uploaded CSV file (a Streamlit `UploadedFile`, or any equivalent
file-like object) into a pandas DataFrame.

This module is responsible only for safely loading and normalizing CSV data.
It does not profile columns, touch DuckDB, or execute SQL.

Design notes
------------
- Size limits reuse `config.max_upload_size_mb`.
- Two size checks are performed:
    1. A fast pre-check using `.size` when available.
    2. A defensive post-read check using the actual byte count.
- Encoding is attempted as `utf-8-sig` first, then `cp1252`.
- CSV column names are normalized after loading:
    - UTF-8 non-breaking spaces are converted to regular spaces.
    - Repeated whitespace is collapsed.
    - Leading/trailing whitespace is removed.
- This normalization is important because CSV files exported from
  spreadsheets/web pages can contain visually invisible Unicode whitespace
  characters that cause SQL column-reference failures.
- Pandas-level exceptions are translated into this module's own exception
  hierarchy.
"""

import io
import re
from typing import Optional

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from app.config import config
from app.utils.logger import get_logger


logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class DataLoaderError(Exception):
    """Base class for all errors raised while loading a CSV file."""


class EmptyFileError(DataLoaderError):
    """The uploaded file has no content, or no parseable rows/columns."""


class FileTooLargeError(DataLoaderError):
    """The uploaded file exceeds the configured maximum upload size."""


class CSVEncodingError(DataLoaderError):
    """The file's text encoding could not be determined/decoded."""


class CSVParsingError(DataLoaderError):
    """The file's content could not be parsed as valid CSV."""


# --------------------------------------------------------------------------
# Encoding configuration
# --------------------------------------------------------------------------

# utf-8-sig handles both:
#   - ordinary UTF-8
#   - UTF-8 files containing a BOM
#
# cp1252 is a common fallback for CSV files exported from Excel on Windows.
_ENCODING_FALLBACKS = (
    "utf-8-sig",
    "cp1252",
)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def load_csv(
    uploaded_file,
    max_size_mb: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    uploaded_file:
        A Streamlit `UploadedFile`, or any file-like object exposing
        `.read()` and ideally `.seek()`, `.name`, and `.size`.

    max_size_mb:
        Optional override for the maximum allowed file size in megabytes.
        Defaults to `config.max_upload_size_mb`.

    Returns
    -------
    pandas.DataFrame
        The loaded and normalized DataFrame.

    Raises
    ------
    ValueError
        If `uploaded_file` is None.

    FileTooLargeError
        If the file exceeds the configured size limit.

    EmptyFileError
        If the file is empty, or has no parseable columns.

    CSVEncodingError
        If the file cannot be decoded using the supported encodings.

    CSVParsingError
        If the file cannot be parsed as valid CSV.
    """

    # ----------------------------------------------------------------------
    # Validate input
    # ----------------------------------------------------------------------

    if uploaded_file is None:
        raise ValueError("uploaded_file must not be None")

    filename = getattr(
        uploaded_file,
        "name",
        "<uploaded file>",
    )

    # Resolve configured or explicitly supplied size limit.
    limit_mb = (
        config.max_upload_size_mb
        if max_size_mb is None
        else max_size_mb
    )

    limit_bytes = limit_mb * 1024 * 1024

    # ----------------------------------------------------------------------
    # Fast size check
    # ----------------------------------------------------------------------

    # Streamlit UploadedFile exposes `.size`, so oversized uploads can be
    # rejected before reading the entire file into memory.
    declared_size = getattr(
        uploaded_file,
        "size",
        None,
    )

    if declared_size is not None and declared_size > limit_bytes:
        _raise_too_large(
            filename=filename,
            size_bytes=declared_size,
            limit_mb=limit_mb,
        )

    # ----------------------------------------------------------------------
    # Read file
    # ----------------------------------------------------------------------

    raw_bytes = _read_all_bytes(uploaded_file)

    # ----------------------------------------------------------------------
    # Empty file check
    # ----------------------------------------------------------------------

    if len(raw_bytes) == 0:
        logger.warning(
            "Rejected empty upload | filename=%s",
            filename,
        )

        raise EmptyFileError(
            f"'{filename}' is empty (0 bytes). "
            "Please upload a non-empty CSV file."
        )

    # ----------------------------------------------------------------------
    # Defensive size check
    # ----------------------------------------------------------------------

    # Some file-like objects do not expose `.size`, or may report an
    # inaccurate size. Always check the actual bytes read as well.
    if len(raw_bytes) > limit_bytes:
        _raise_too_large(
            filename=filename,
            size_bytes=len(raw_bytes),
            limit_mb=limit_mb,
        )

    # ----------------------------------------------------------------------
    # Parse CSV
    # ----------------------------------------------------------------------

    df = _parse_csv_bytes(
        raw_bytes=raw_bytes,
        filename=filename,
    )

    # ----------------------------------------------------------------------
    # Normalize column names
    # ----------------------------------------------------------------------

    df = _normalize_column_names(df)

    # ----------------------------------------------------------------------
    # Logging
    # ----------------------------------------------------------------------

    size_mb = len(raw_bytes) / (1024 * 1024)

    logger.info(
        "Loaded CSV | filename=%s | rows=%d | columns=%d | size_mb=%.2f",
        filename,
        len(df),
        len(df.columns),
        size_mb,
    )

    if df.empty:
        logger.warning(
            "Loaded CSV has zero data rows | filename=%s",
            filename,
        )

    return df


# --------------------------------------------------------------------------
# File size handling
# --------------------------------------------------------------------------


def _raise_too_large(
    filename: str,
    size_bytes: float,
    limit_mb: float,
) -> None:
    """
    Raise a standardized FileTooLargeError.
    """

    size_mb = size_bytes / (1024 * 1024)

    logger.warning(
        "Rejected oversized upload | filename=%s | size_mb=%.2f | limit_mb=%s",
        filename,
        size_mb,
        limit_mb,
    )

    raise FileTooLargeError(
        f"'{filename}' is {size_mb:.1f} MB, which exceeds "
        f"the {limit_mb} MB upload limit."
    )


# --------------------------------------------------------------------------
# File reading
# --------------------------------------------------------------------------


def _read_all_bytes(uploaded_file) -> bytes:
    """
    Read the full contents of a file-like object as bytes.
    """

    # Reset the file position when possible so callers can safely upload
    # a file that has previously been read.
    if hasattr(uploaded_file, "seek"):
        try:
            uploaded_file.seek(0)
        except (OSError, ValueError):
            # Not seekable; continue from the current position.
            pass

    try:
        data = uploaded_file.read()

    except Exception as exc:
        raise DataLoaderError(
            f"Could not read the uploaded file: {exc}"
        ) from exc

    # Some file-like objects opened in text mode return str.
    if isinstance(data, str):
        data = data.encode("utf-8")

    if not isinstance(data, bytes):
        raise DataLoaderError(
            "Uploaded file reader returned an unsupported data type."
        )

    return data


# --------------------------------------------------------------------------
# CSV parsing
# --------------------------------------------------------------------------


def _parse_csv_bytes(
    raw_bytes: bytes,
    filename: str,
) -> pd.DataFrame:
    """
    Parse CSV bytes into a DataFrame.

    Encoding order:
        1. utf-8-sig
        2. cp1252
    """

    last_error: Optional[UnicodeDecodeError] = None

    for encoding in _ENCODING_FALLBACKS:

        try:
            df = pd.read_csv(
                io.BytesIO(raw_bytes),
                encoding=encoding,
            )

        except UnicodeDecodeError as exc:
            last_error = exc
            continue

        except EmptyDataError as exc:
            raise EmptyFileError(
                f"'{filename}' has no columns to parse "
                "(empty or header-less CSV)."
            ) from exc

        except ParserError as exc:
            raise CSVParsingError(
                f"'{filename}' could not be parsed as CSV: {exc}"
            ) from exc

        except Exception as exc:
            raise CSVParsingError(
                f"Unexpected error parsing '{filename}': {exc}"
            ) from exc

        else:

            if encoding != _ENCODING_FALLBACKS[0]:
                logger.warning(
                    "Parsed '%s' using fallback encoding=%s "
                    "after the default encoding failed",
                    filename,
                    encoding,
                )

            return df

    raise CSVEncodingError(
        f"'{filename}' could not be decoded with any supported encoding "
        f"({', '.join(_ENCODING_FALLBACKS)}). "
        "Try re-saving it as UTF-8."
    ) from last_error


# --------------------------------------------------------------------------
# Column-name normalization
# --------------------------------------------------------------------------


def _normalize_column_names(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize CSV column names without changing their meaning.

    This fixes common invisible/Unicode whitespace problems found in
    CSV files exported from Excel, web pages, and other spreadsheet tools.

    Examples
    --------
    ``"Actual\\u00A0gross"``
        becomes
    ``"Actual gross"``

    ``"  Artist  "``
        becomes
    ``"Artist"``

    ``"Adjusted   gross"``
        becomes
    ``"Adjusted gross"``

    The operation does not modify the underlying data values.
    """

    normalized_columns = []

    for column in df.columns:

        # Preserve non-string column labels safely.
        column_text = str(column)

        # Replace common Unicode whitespace characters with ordinary
        # spaces. In particular, \u00A0 is a non-breaking space.
        column_text = (
            column_text
            .replace("\u00A0", " ")
            .replace("\u2007", " ")
            .replace("\u202F", " ")
        )

        # Normalize any remaining whitespace sequences.
        column_text = re.sub(
            r"\s+",
            " ",
            column_text,
        )

        # Remove leading/trailing whitespace.
        column_text = column_text.strip()

        normalized_columns.append(column_text)

    # Log only when normalization actually changes a column name.
    original_columns = [str(column) for column in df.columns]

    if normalized_columns != original_columns:
        changes = [
            f"{old!r} -> {new!r}"
            for old, new in zip(
                original_columns,
                normalized_columns,
            )
            if old != new
        ]

        logger.info(
            "Normalized CSV column names | changes=%s",
            changes,
        )

    # Work on the DataFrame's columns rather than mutating cell data.
    df = df.copy()
    df.columns = normalized_columns

    return df