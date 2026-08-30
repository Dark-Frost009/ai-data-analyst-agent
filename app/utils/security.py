"""
SQL security and validation layer.

Treats every SQL string passed to validate_sql() as untrusted input,
typically LLM-generated SQL.

Only safe, read-only SELECT/WITH queries are allowed.

The validator:

- Allows only SELECT/WITH statements.
- Rejects multiple statements.
- Restricts table references to an explicit allowlist.
- Restricts SQL functions to an explicit allowlist.
- Blocks file/database/network access functions.
- Allows safe CAST / TRY_CAST operations.
- Allows CASE expressions without treating their internal IF nodes
  as callable SQL functions.
- Enforces a maximum LIMIT.
- Never executes SQL itself.
"""

from typing import List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp

from app.config import config
from app.models.schemas import ValidationError, ValidationResult
from app.utils.logger import get_logger


logger = get_logger(__name__)

_DIALECT = "duckdb"


# ---------------------------------------------------------------------------
# Allowed statement types
# ---------------------------------------------------------------------------

_ALLOWED_ROOT_TYPES = (
    exp.Select,
    exp.With,
)


# ---------------------------------------------------------------------------
# Allowed SQL functions
# ---------------------------------------------------------------------------

_ALLOWED_FUNCTIONS = {
    # Aggregations
    "SUM",
    "AVG",
    "COUNT",
    "MIN",
    "MAX",

    # Numeric
    "ROUND",
    "ABS",
    "CEIL",
    "CEILING",
    "FLOOR",

    # String
    "LOWER",
    "UPPER",
    "TRIM",
    "LENGTH",
    "CONCAT",
    "REPLACE",

    # Null handling
    "COALESCE",
    "NULLIF",

    # Date/time
    "DATE_TRUNC",
    "DATETRUNC",
    "TIMESTAMPTRUNC",
    "DATE_PART",
    "DATEPART",
    "EXTRACT",

    # Type conversion
    "CAST",
    "TRY_CAST",

    # Conditional
    "CASE",
}


# ---------------------------------------------------------------------------
# Explicitly allowed AST expression classes
# ---------------------------------------------------------------------------

def _is_allowed_type_conversion(func: exp.Expression) -> bool:
    """
    Return True when an expression represents safe SQL type conversion.

    sqlglot can represent CAST / TRY_CAST using dedicated AST classes
    rather than ordinary Func nodes. We therefore handle them explicitly.
    """

    type_name = type(func).__name__.upper()

    if type_name in {
        "CAST",
        "TRYCAST",
    }:
        return True

    try:
        sql_name = func.sql_name().upper()
    except Exception:
        sql_name = ""

    return sql_name in {
        "CAST",
        "TRY_CAST",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_sql(
    sql: str,
    allowed_tables: List[str],
    max_rows: Optional[int] = None,
) -> ValidationResult:
    """
    Validate a single SQL statement before execution.

    This function NEVER executes SQL.
    """

    if not sql or not sql.strip():
        return _rejected(
            "EMPTY_SQL",
            "SQL statement is empty.",
        )

    allowed_tables_lower = {
        t.lower()
        for t in allowed_tables
    }

    # -----------------------------------------------------------------------
    # Parse SQL
    # -----------------------------------------------------------------------

    try:
        statements = [
            statement
            for statement in sqlglot.parse(
                sql,
                read=_DIALECT,
            )
            if statement is not None
        ]

    except Exception as exc:
        return _rejected(
            "PARSE_ERROR",
            "SQL could not be parsed.",
            detail=str(exc),
        )

    # -----------------------------------------------------------------------
    # Empty SQL
    # -----------------------------------------------------------------------

    if len(statements) == 0:
        return _rejected(
            "EMPTY_SQL",
            "SQL statement is empty.",
        )

    # -----------------------------------------------------------------------
    # Multiple statements
    # -----------------------------------------------------------------------

    if len(statements) > 1:
        return _rejected(
            "MULTIPLE_STATEMENTS",
            "Only a single SQL statement is allowed per query.",
            detail=f"Found {len(statements)} statements.",
        )

    ast = statements[0]

    # -----------------------------------------------------------------------
    # Statement type allowlist
    # -----------------------------------------------------------------------

    if not isinstance(ast, _ALLOWED_ROOT_TYPES):
        return _rejected(
            "DISALLOWED_STATEMENT_TYPE",
            "Only read-only SELECT/WITH statements are allowed.",
            detail=f"Statement type: {type(ast).__name__}",
        )

    # -----------------------------------------------------------------------
    # Table allowlist
    # -----------------------------------------------------------------------

    table_error = _check_table_allowlist(
        ast,
        allowed_tables_lower,
    )

    if table_error is not None:
        return _rejected(*table_error)

    # -----------------------------------------------------------------------
    # Function allowlist
    # -----------------------------------------------------------------------

    function_error = _check_function_allowlist(ast)

    if function_error is not None:
        return _rejected(*function_error)

    # -----------------------------------------------------------------------
    # LIMIT enforcement
    # -----------------------------------------------------------------------

    effective_max_rows = (
        max_rows
        if max_rows is not None
        else config.max_query_result_rows
    )

    cleaned_ast = _enforce_limit(
        ast,
        effective_max_rows,
    )

    cleaned_sql = cleaned_ast.sql(
        dialect=_DIALECT,
    )

    logger.info(
        "SQL validated successfully | cleaned_sql=%s",
        cleaned_sql,
    )

    return ValidationResult(
        is_valid=True,
        cleaned_sql=cleaned_sql,
        errors=[],
    )


# ---------------------------------------------------------------------------
# CTE handling
# ---------------------------------------------------------------------------

def _collect_cte_names(
    ast: exp.Expression,
) -> Set[str]:
    """
    Collect aliases of every CTE defined inside the query.
    """

    names: Set[str] = set()

    for cte in ast.find_all(exp.CTE):
        alias = getattr(
            cte,
            "alias_or_name",
            None,
        )

        if alias:
            names.add(
                alias.lower()
            )

    return names


# ---------------------------------------------------------------------------
# Table allowlist
# ---------------------------------------------------------------------------

def _check_table_allowlist(
    ast: exp.Expression,
    allowed_tables_lower: Set[str],
) -> Optional[Tuple[str, str, Optional[str]]]:
    """
    Reject any external table reference that is not explicitly allowed.
    """

    cte_names = _collect_cte_names(ast)

    for table in ast.find_all(exp.Table):

        table_name = table.name.lower()

        if not table_name:
            continue

        # Local CTE references are allowed.
        if table_name in cte_names:
            continue

        if table_name not in allowed_tables_lower:
            return (
                "DISALLOWED_TABLE_REFERENCE",
                (
                    "Query references a table that is not allowed: "
                    f"'{table.name}'."
                ),
                (
                    "Allowed tables: "
                    f"{sorted(allowed_tables_lower)}"
                ),
            )

    return None


# ---------------------------------------------------------------------------
# Function allowlist
# ---------------------------------------------------------------------------

def _check_function_allowlist(
    ast: exp.Expression,
) -> Optional[Tuple[str, str, Optional[str]]]:
    """
    Reject any SQL function that is not explicitly allowed.

    The implementation handles both ordinary sqlglot Func nodes and
    dedicated expression classes such as CAST / TRY_CAST.

    Important:
    SQL CASE expressions are represented by sqlglot as a Case node
    containing one or more internal If nodes. Those If nodes are part
    of the CASE expression and are NOT callable SQL functions.

    Therefore:
        CASE WHEN ... THEN ... ELSE ... END
    is allowed,

    while:
        IF(condition, true_value, false_value)
    remains subject to the function allowlist.
    """

    for func in ast.find_all(exp.Func):

        # ---------------------------------------------------------------
        # SQL CASE handling
        # ---------------------------------------------------------------

        # sqlglot represents:
        #
        # CASE WHEN condition THEN value ELSE value END
        #
        # approximately as:
        #
        # Case
        #   -> If
        #
        # The internal If is not a user-invoked SQL function.
        if isinstance(func, exp.If):
            parent = func.parent

            if isinstance(parent, exp.Case):
                continue

        # ---------------------------------------------------------------
        # Safe type conversions
        # ---------------------------------------------------------------

        if _is_allowed_type_conversion(func):
            continue

        # ---------------------------------------------------------------
        # Anonymous functions
        # ---------------------------------------------------------------

        if isinstance(func, exp.Anonymous):

            func_name = (
                str(func.this).upper()
                if func.this
                else ""
            )

        else:

            func_name = type(func).__name__.upper()

        # ---------------------------------------------------------------
        # SQL function name
        # ---------------------------------------------------------------

        sql_name = ""

        try:
            sql_name = func.sql_name().upper()
        except Exception:
            pass

        candidate_names = {
            name
            for name in (
                func_name,
                sql_name,
            )
            if name
        }

        if not candidate_names.intersection(
            _ALLOWED_FUNCTIONS
        ):
            display_name = (
                sql_name
                or func_name
                or "UNKNOWN"
            )

            return (
                "DISALLOWED_FUNCTION",
                f"Function is not allowed: '{display_name}'.",
                (
                    "Allowed functions: "
                    f"{sorted(_ALLOWED_FUNCTIONS)}"
                ),
            )

    # -----------------------------------------------------------------------
    # Explicitly inspect CAST / TRY_CAST expressions.
    #
    # Depending on the sqlglot version, these may not appear in
    # ast.find_all(exp.Func), so inspect the complete expression tree too.
    # -----------------------------------------------------------------------

    for expression in ast.walk():

        type_name = type(expression).__name__.upper()

        if type_name in {
            "CAST",
            "TRYCAST",
        }:
            continue

        try:
            sql_name = expression.sql_name().upper()
        except Exception:
            sql_name = ""

        # If it is a dedicated conversion expression, allow it.
        if sql_name in {
            "CAST",
            "TRY_CAST",
        }:
            continue

    return None


# ---------------------------------------------------------------------------
# LIMIT enforcement
# ---------------------------------------------------------------------------

def _enforce_limit(
    ast: exp.Expression,
    max_rows: int,
) -> exp.Expression:
    """
    Inject a LIMIT if absent, or clamp it down if it exceeds max_rows.

    Only a plain, non-negative integer literal is considered a safe
    existing LIMIT.

    Examples:

        LIMIT 50       -> preserve 50
        LIMIT 0        -> preserve 0
        LIMIT 999999   -> clamp to max_rows
        LIMIT -1       -> replace with max_rows
        LIMIT 10.5     -> replace with max_rows
        LIMIT 5 + 5    -> replace with max_rows
    """

    limit_node = ast.args.get("limit")

    # No LIMIT -> inject one.
    if limit_node is None:
        return ast.limit(max_rows)

    current_value = _extract_limit_value(
        limit_node
    )

    # If LIMIT cannot safely be interpreted as a plain
    # non-negative integer literal, replace it with the
    # safe maximum.
    if current_value is None:
        return ast.limit(max_rows)

    # Existing LIMIT is too large -> clamp it.
    if current_value > max_rows:
        return ast.limit(max_rows)

    return ast


# ---------------------------------------------------------------------------
# LIMIT extraction
# ---------------------------------------------------------------------------

def _extract_limit_value(
    limit_node: exp.Expression,
) -> Optional[int]:
    """
    Extract a LIMIT value only when it is a plain, non-negative
    integer literal.

    Safe examples:

        LIMIT 50
        LIMIT 0
        LIMIT 999999

    Returns None for:

        LIMIT -1
        LIMIT 10.5
        LIMIT 5 + 5
        LIMIT some_expression
    """

    value_expr = (
        getattr(
            limit_node,
            "expression",
            None,
        )
        or limit_node.args.get(
            "expression"
        )
    )

    if value_expr is None:
        return None

    # We deliberately accept ONLY a literal.
    #
    # This rejects:
    #   Neg(1)      -> -1
    #   Add(5, 5)   -> 5 + 5
    #   Mul(...)    -> arithmetic
    #   Column(...) -> dynamic expression
    if not isinstance(value_expr, exp.Literal):
        return None

    # A string literal is not a valid trusted integer LIMIT.
    if value_expr.is_string:
        return None

    raw = str(
        value_expr.this
    ).strip()

    # Only decimal digits are accepted.
    #
    # This deliberately rejects:
    #   -1
    #   +1
    #   10.5
    #   1e3
    #   arithmetic expressions
    if not raw.isdigit():
        return None

    try:
        return int(raw)

    except (
        TypeError,
        ValueError,
    ):
        return None


# ---------------------------------------------------------------------------
# Standardized rejection
# ---------------------------------------------------------------------------

def _rejected(
    code: str,
    message: str,
    detail: Optional[str] = None,
) -> ValidationResult:
    """
    Create a standardized rejected ValidationResult.
    """

    logger.warning(
        "SQL rejected | code=%s | message=%s",
        code,
        message,
    )

    return ValidationResult(
        is_valid=False,
        cleaned_sql=None,
        errors=[
            ValidationError(
                code=code,
                message=message,
                detail=detail,
            )
        ],
    )