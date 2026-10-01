"""Check conversion fidelity after SQL security validation, without executing SQL."""
from decimal import Decimal, InvalidOperation, localcontext

import pandas as pd
import sqlglot
from sqlglot import exp

from app.core.numeric_preparation import can_represent_number


class SemanticConversionError(Exception):
    """A planned conversion is ambiguous, invalid or would lose precision."""


def _values(expression, dataframe, physical_columns):
    """Evaluate only a small set of pure, per-value cleaning expressions."""
    if isinstance(expression, exp.Column):
        if id(expression) not in physical_columns:
            return None
        # Only physical dataset columns; derived/CTE aliases are not guessed.
        names = {str(name).casefold(): name for name in dataframe.columns}
        name = names.get(expression.name.casefold())
        return dataframe[name] if name is not None else None
    if isinstance(expression, exp.Literal):
        return pd.Series([expression.this] * len(dataframe), index=dataframe.index)
    if isinstance(expression, exp.Paren):
        return _values(expression.this, dataframe, physical_columns)
    if isinstance(expression, exp.Cast) and expression.to.this in (exp.DataType.Type.VARCHAR, exp.DataType.Type.TEXT):
        values = _values(expression.this, dataframe, physical_columns)
        return values.astype("string") if values is not None else None
    if isinstance(expression, exp.Replace):
        values = _values(expression.this, dataframe, physical_columns)
        old, new = expression.args.get("expression"), expression.args.get("replacement")
        if values is not None and isinstance(old, exp.Literal) and isinstance(new, exp.Literal):
            return values.astype("string").str.replace(str(old.this), str(new.this), regex=False)
    if isinstance(expression, exp.Trim) and not expression.args.get("expression"):
        values = _values(expression.this, dataframe, physical_columns)
        return values.astype("string").str.strip() if values is not None else None
    return None


def check_conversion_fidelity(sql, dataframe):
    tree = sqlglot.parse_one(sql, read="duckdb")
    from sqlglot.optimizer.scope import traverse_scope
    physical_columns = set()
    for scope in traverse_scope(tree):
        selected = {alias: source for alias, (_, source) in scope.selected_sources.items()}
        for column in scope.columns:
            source = selected.get(column.table) if column.table else next(iter(selected.values()), None) if len(selected) == 1 else None
            if isinstance(source, exp.Table) and source.name.casefold() == "dataset":
                physical_columns.add(id(column))
    temporal = {exp.DataType.Type.DATE, exp.DataType.Type.TIMESTAMP, exp.DataType.Type.TIMESTAMPTZ}
    integers = {exp.DataType.Type.TINYINT: 8, exp.DataType.Type.SMALLINT: 16,
                exp.DataType.Type.INT: 32, exp.DataType.Type.BIGINT: 64, exp.DataType.Type.INT128: 128}
    unsigned = {exp.DataType.Type.UTINYINT: 8, exp.DataType.Type.USMALLINT: 16,
                exp.DataType.Type.UINT: 32, exp.DataType.Type.UBIGINT: 64, exp.DataType.Type.UINT128: 128}
    floats = {exp.DataType.Type.FLOAT, exp.DataType.Type.DOUBLE}
    numeric = set(integers) | set(unsigned) | floats | {exp.DataType.Type.DECIMAL}
    for cast in tree.find_all(exp.Cast):
        kind = cast.to.this
        if kind not in temporal | numeric:
            continue
        values = _values(cast.this, dataframe, physical_columns)
        if values is None:
            # A cast of an aggregate or derived alias cannot be verified per source row.
            raise SemanticConversionError("This conversion cannot be checked safely. Use the column's prepared type directly.")
        non_null = values.dropna()
        if kind in temporal:
            if isinstance(cast.this, exp.Literal):
                from app.core.execution_preparation import _coerce_datetime
                if _coerce_datetime(pd.Series([cast.this.this])) is not None:
                    continue
            if not pd.api.types.is_datetime64_any_dtype(values.dtype):
                raise SemanticConversionError(
                    "Date conversion was stopped because the column remains text or numeric. "
                    "Provide consistently formatted, unambiguous calendar dates before filtering by date."
                )
            continue
        # Do not turn dates or booleans into numeric units implicitly.
        if pd.api.types.is_datetime64_any_dtype(values.dtype) or pd.api.types.is_bool_dtype(values.dtype):
            raise SemanticConversionError("This numeric conversion changes the column's meaning.")
        valid = True
        for value in non_null:
            if kind in integers or kind in unsigned:
                bits = integers.get(kind, unsigned.get(kind))
                try:
                    number = Decimal(str(value))
                    lower = 0 if kind in unsigned else -(2**(bits-1))
                    upper = 2**bits if kind in unsigned else 2**(bits-1)
                    valid = number.is_finite() and number == number.to_integral_value() and lower <= number < upper
                except InvalidOperation:
                    valid = False
            elif kind in floats:
                valid = can_represent_number(value)
                if valid and kind == exp.DataType.Type.FLOAT:
                    import numpy as np
                    valid = Decimal(str(float(np.float32(value)))) == Decimal(str(value))
            else:
                parameters = cast.to.expressions
                precision = int(parameters[0].this.this) if parameters else 18
                scale = int(parameters[1].this.this) if len(parameters) > 1 else 3
                try:
                    number = Decimal(str(value))
                    with localcontext() as context:
                        context.prec = max(precision, len(number.as_tuple().digits) + scale + 1)
                        valid = number.is_finite() and number == number.quantize(Decimal(1).scaleb(-scale)) and abs(number) < Decimal(10) ** (precision-scale)
                except (InvalidOperation, ValueError):
                    valid = False
            if not valid:
                break
        if not valid:
            raise SemanticConversionError(
                "Numeric conversion was stopped because values would be discarded, rounded or exceed the target type. "
                "Clean the column or use an exact numeric representation."
            )
