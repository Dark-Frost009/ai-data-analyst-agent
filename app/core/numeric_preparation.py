"""Lossless canonical numeric conversion, separate from faithful CSV loading."""
from decimal import Decimal, InvalidOperation
import math
import re

import pandas as pd


def _decimal(value):
    try:
        number = Decimal(str(value))
        return number if number.is_finite() else None
    except InvalidOperation:
        return None


def can_represent_number(value, kind="double"):
    number = _decimal(value)
    if number is None:
        return False
    if kind == "integer":
        return number == number.to_integral_value() and -(2**63) <= number < 2**63
    try:
        converted = float(number)
        # Integers outside exact binary precision must not be rounded into floats.
        if number == number.to_integral_value() and Decimal.from_float(converted) != number:
            return False
        return math.isfinite(converted) and Decimal(str(converted)) == number
    except (ValueError, OverflowError):
        return False


def coerce_numeric(series):
    values = series.dropna()
    if values.empty or not all(isinstance(v, str) for v in values):
        return None
    if all(re.fullmatch(r"[0-9]{4}", v) for v in values):
        # Keep bare-year columns unchanged; this is not a calendar-date coercion.
        return None
    # Preserve identifiers with zero padding and unsupported separators/units.
    pattern = r"[+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
    if not all(re.fullmatch(pattern, v) for v in values):
        return None
    if all(re.fullmatch(r"[+-]?(?:0|[1-9][0-9]*)", v) for v in values):
        if not all(can_represent_number(v, "integer") for v in values):
            return None
        return pd.Series(pd.array([None if pd.isna(v) else int(v) for v in series], dtype="Int64"), index=series.index, name=series.name)
    if not all(can_represent_number(v) for v in values):
        return None
    return pd.Series(pd.array([None if pd.isna(v) else float(v) for v in series], dtype="Float64"), index=series.index, name=series.name)
