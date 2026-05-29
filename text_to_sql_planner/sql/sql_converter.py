"""SQL converter: transforms an OperationTree into a SQL SELECT statement.

Walks the operation tree depth-first to build a _QueryContext with globally
unique table aliases and variable-to-alias mappings, then renders flat SQL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Union

from text_to_sql_planner.types.operation_tree import (
    OperationTree,
    OperationNode,
    TableLeafNode,
    OperatorNode,
)
from text_to_sql_planner.types.operators import (
    SelectionParams,
    JoinParams,
    ProjectionParams,
    CartesianProductParams,
    UnionParams,
    DivisionParams,
)
from text_to_sql_planner.types.drc import (
    ComparisonNode,
    LogicalConnectiveNode,
    NotNode,
    VariableRefNode,
    LiteralNode,
    MembershipNode,
    QuantifierNode,
    FunctionCallNode,
    DRCCondition,
    ColumnVariable,
    AggregateVariable,
)


# --- Result types ---


@dataclass
class SQLSuccess:
    sql: str


@dataclass
class SQLFailure:
    error: str


SQLResult = Union[SQLSuccess, SQLFailure]


# --- Constants and helpers ---

_AGGREGATE_FUNCS = {"COUNT", "SUM", "AVG", "MIN", "MAX"}


def _is_date_string(s: str) -> bool:
    """Check if a string looks like a date (YYYY-MM-DD or YYYY/MM/DD)."""
    return bool(re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}$", s))


def _days_to_interval(days_str: str) -> str:
    """Convert a number of days to the most readable interval unit.

    Examples:
        "10950" -> "30 years"
        "365" -> "1 year"
        "730" -> "2 years"
        "30" -> "30 days"
        "90" -> "3 months"
    """
    try:
        days = int(days_str)
    except (ValueError, TypeError):
        return f"{days_str} days"

    if days % 365 == 0:
        years = days // 365
        return "1 year" if years == 1 else f"{years} years"
    if days % 30 == 0 and days < 365:
        months = days // 30
        return "1 month" if months == 1 else f"{months} months"
    if days % 7 == 0 and days < 30:
        weeks = days // 7
        return "1 week" if weeks == 1 else f"{weeks} weeks"
    return "1 day" if days == 1 else f"{days} days"


def _flip_operator(op: str) -> str:
    """Flip a comparison operator (e.g., >= becomes <=)."""
    flips = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "=": "=", "!=": "!="}
    return flips.get(op, op)


def _strip_suffix(name: str) -> tuple[str, str]:
    """Strip DRC suffixes (_r2, _1, _2) and return (base_name, suffix)."""
    if name.endswith("_r2") and len(name) > 3:
        base = name[:-3]
        if base and not base.endswith("_"):
            return base, "_r2"
    if len(name) > 2 and name[-2] == "_" and name[-1] in "123456789":
        base = name[:-2]
        if base and not base.endswith("_"):
            return base, name[-2:]
    return name, ""


# --- Conversion error ---


class _ConversionError(Exception):
    pass


# --- Query context ---


@dataclass
class _QueryContext:
    """Tracks table aliases and variable mappings during SQL generation."""
    # Maps table_name -> list of aliases (for self-joins: ["p1", "p2"])
    table_instances: dict[str, list[str]] = field(default_factory=dict)
    # Maps DRC variable name -> "alias.column" (e.g., "emp_id_1" -> "p1.emp_id")
    var_mapping: dict[str, str] = field(default_factory=dict)
    # All FROM/JOIN clauses
    from_clause: str = ""
    join_clauses: list[str] = field(default_factory=list)
    where_conditions: list[str] = field(default_factory=list)


# --- Alias generator ---


class _AliasGenerator:
    """Generates globally unique short aliases for table instances."""

    def __init__(self):
        self._counters: dict[str, int] = {}

    def next_alias(self, table_name: str) -> str:
        """Generate next unique alias for a table.

        Uses first letter lowercase + counter: e1, e2, p1, p2, d1, etc.
        """
        prefix = table_name[0].lower() if table_name else "t"
        count = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = count
        return f"{prefix}{count}"


# --- SQL Generator ---


class _SqlGenerator:
    """Generates SQL from an OperationTree with globally unique aliases."""

    def __init__(self):
        self._aliases = _AliasGenerator()

    def generate(
        self,
        tree: OperationTree,
        result_variables: list | None = None,
        distinct: bool = False,
        order_by: list | None = None,
        limit: int | None = None,
    ) -> SQLResult:
        """Generate SQL from an operation tree."""
        if tree is None or tree.root is None:
            return SQLFailure(error="Invalid operation tree: root is None")
        try:
            if result_variables:
                return self._generate_with_result_vars(
                    tree,
                    result_variables,
                    distinct=distinct,
                    order_by=order_by,
                    limit=limit,
                )
            sql = self._convert_node(tree.root)
            return SQLSuccess(sql=sql)
        except _ConversionError as e:
            return SQLFailure(error=str(e))

    # --- Top-level node dispatch ---

    def _convert_node(self, node: OperationNode) -> str:
        """Convert an operation node to SQL string."""
        if node is None:
            raise _ConversionError("Invalid tree: encountered None node")
        if isinstance(node, TableLeafNode):
            return self._convert_table_leaf(node)
        elif isinstance(node, OperatorNode):
            return self._convert_operator(node)
        raise _ConversionError(f"Unknown node type: {type(node).__name__}")

    def _convert_table_leaf(self, node: TableLeafNode) -> str:
        """Table leaf -> SELECT alias.cols FROM table alias."""
        if not node.table_name:
            raise _ConversionError("Table leaf has empty table name")
        alias = self._aliases.next_alias(node.table_name)
        if node.columns:
            cols = ", ".join(f"{alias}.{c}" for c in node.columns)
            return f"SELECT {cols} FROM {node.table_name} {alias}"
        return f"SELECT * FROM {node.table_name} {alias}"

    def _convert_operator(self, node: OperatorNode) -> str:
        """Route operator to the appropriate handler."""
        params = node.params
        if isinstance(params, UnionParams):
            return self._convert_union(node)
        elif isinstance(params, DivisionParams):
            return self._convert_division(node)
        elif isinstance(params, ProjectionParams):
            return self._convert_projection(node, params)
        else:
            # Selection, Join, CartesianProduct -> flatten
            ctx = self._build_context(node)
            return self._render_context(ctx, node)

    # --- Context building (depth-first walk) ---

    def _build_context(self, node: OperationNode) -> _QueryContext:
        """Walk the operation tree depth-first to build a flat query context."""
        if isinstance(node, TableLeafNode):
            return self._ctx_from_table(node)
        if not isinstance(node, OperatorNode):
            raise _ConversionError(f"Cannot build context for: {type(node).__name__}")

        params = node.params
        inputs = node.inputs

        if isinstance(params, SelectionParams):
            return self._ctx_selection(node, params, inputs)
        elif isinstance(params, JoinParams):
            return self._ctx_join(node, params, inputs)
        elif isinstance(params, CartesianProductParams):
            return self._ctx_cartesian(node, params, inputs)
        elif isinstance(params, (ProjectionParams, UnionParams, DivisionParams)):
            # Non-flattenable operators get wrapped as a derived subquery
            return self._ctx_from_subquery(node)
        else:
            raise _ConversionError(f"Cannot flatten operator: {node.operator}")

    def _ctx_from_table(self, node: TableLeafNode) -> _QueryContext:
        """Build context for a table leaf."""
        if not node.table_name:
            raise _ConversionError("Table leaf has empty table name")
        alias = self._aliases.next_alias(node.table_name)
        ctx = _QueryContext()
        ctx.from_clause = f"{node.table_name} {alias}"
        ctx.table_instances[node.table_name] = [alias]
        for col in node.columns:
            ctx.var_mapping[col] = f"{alias}.{col}"
        return ctx

    def _ctx_from_subquery(self, node: OperatorNode) -> _QueryContext:
        """Wrap a non-flattenable operator (projection/union/division) as a
        derived subquery and expose its output columns through a fresh alias.

        Output columns of the inner operator become bare column names in the
        derived table, qualified with the new alias.
        """
        inner_sql = self._convert_node(node)
        # Pick a generic alias prefix — use 's' for "subquery"
        alias = self._aliases.next_alias("subquery")

        cols = self._get_node_columns(node)
        if not cols:
            # Best effort: emit alias without column mapping. Outer references
            # will fall back via _resolve_var.
            cols = []

        ctx = _QueryContext()
        indented = self._indent_sql(inner_sql, 4)
        ctx.from_clause = f"(\n{indented}\n  ) {alias}"
        # Track this as a "synthetic" table instance for later merging
        ctx.table_instances[alias] = [alias]
        for col in cols:
            # Strip aggregate/expression wrappers — only bare column names
            # survive into the derived table's schema.
            bare = self._extract_underlying_column(col)
            ctx.var_mapping[bare] = f"{alias}.{bare}"
        return ctx

    def _ctx_selection(self, node: OperatorNode, params: SelectionParams, inputs: list) -> _QueryContext:
        """Build context for a selection (adds WHERE conditions)."""
        if not inputs:
            raise _ConversionError("Selection requires exactly 1 input")
        if params.condition is None:
            raise _ConversionError("Selection requires a condition")

        ctx = self._build_context(inputs[0])
        cond_sql = self._condition_to_sql(params.condition, ctx.var_mapping)
        if cond_sql:
            ctx.where_conditions.append(cond_sql)
        return ctx

    def _ctx_join(self, node: OperatorNode, params: JoinParams, inputs: list) -> _QueryContext:
        """Build context for a join (adds JOIN clause).

        Mirrors the join operator's renaming contract: join columns are
        shared (resolved to the left side), non-join columns from the right
        side get a ``_r2`` suffix when they collide with a name on the left.
        """
        if len(inputs) < 2:
            raise _ConversionError("Join requires exactly 2 inputs")
        if not params.join_columns:
            raise _ConversionError("Join requires at least one join column")

        left_ctx = self._build_context(inputs[0])
        right_ctx = self._build_context(inputs[1])

        # Build ON clause from the left/right contexts (each side still has
        # its own bare name mapping at this point).
        on_parts: list[str] = []
        for col in params.join_columns:
            left_ref = self._resolve_var(col, left_ctx.var_mapping)
            right_ref = self._resolve_var(col, right_ctx.var_mapping)
            on_parts.append(f"{left_ref} = {right_ref}")
        on_clause = " AND ".join(on_parts)

        # Merge contexts: left is base, right becomes a JOIN
        merged = _QueryContext()
        merged.from_clause = left_ctx.from_clause
        merged.join_clauses = (
            left_ctx.join_clauses
            + right_ctx.join_clauses
            + [f"JOIN {right_ctx.from_clause} ON {on_clause}"]
        )
        merged.where_conditions = left_ctx.where_conditions + right_ctx.where_conditions

        # Merge table instances
        merged.table_instances = dict(left_ctx.table_instances)
        for tbl, aliases in right_ctx.table_instances.items():
            merged.table_instances.setdefault(tbl, []).extend(aliases)

        # Build the merged var_map per the join renaming contract.
        join_set = set(params.join_columns)
        merged.var_mapping = dict(left_ctx.var_mapping)
        for name, qualified in right_ctx.var_mapping.items():
            if name in join_set:
                # Join columns are shared; the left side already provides them.
                continue
            if name in left_ctx.var_mapping:
                # Non-join collision — right side is renamed to <name>_r2.
                merged.var_mapping[f"{name}_r2"] = qualified
            else:
                merged.var_mapping[name] = qualified

        return merged

    def _ctx_cartesian(self, node: OperatorNode, params: CartesianProductParams, inputs: list) -> _QueryContext:
        """Build context for a cartesian product (CROSS JOIN).

        Mirrors the cartesian product operator's renaming contract: when a
        column name appears in both inputs, the left side becomes ``_1`` and
        the right side becomes ``_2``. Non-overlapping columns keep their
        bare names.
        """
        if len(inputs) < 2:
            raise _ConversionError("Cartesian product requires exactly 2 inputs")

        left_ctx = self._build_context(inputs[0])
        right_ctx = self._build_context(inputs[1])

        # Merge contexts: left is base, right becomes CROSS JOIN
        merged = _QueryContext()
        merged.from_clause = left_ctx.from_clause
        merged.join_clauses = (
            left_ctx.join_clauses
            + right_ctx.join_clauses
            + [f"CROSS JOIN {right_ctx.from_clause}"]
        )
        merged.where_conditions = left_ctx.where_conditions + right_ctx.where_conditions

        # Merge table instances
        merged.table_instances = dict(left_ctx.table_instances)
        for tbl, aliases in right_ctx.table_instances.items():
            merged.table_instances.setdefault(tbl, []).extend(aliases)

        # Build the merged var_map per the cartesian renaming contract.
        overlap = set(left_ctx.var_mapping) & set(right_ctx.var_mapping)
        merged.var_mapping = {}
        for name, qualified in left_ctx.var_mapping.items():
            if name in overlap:
                merged.var_mapping[f"{name}_1"] = qualified
            else:
                merged.var_mapping[name] = qualified
        for name, qualified in right_ctx.var_mapping.items():
            if name in overlap:
                merged.var_mapping[f"{name}_2"] = qualified
            else:
                merged.var_mapping[name] = qualified

        return merged

    # --- Rendering ---

    def _render_context(self, ctx: _QueryContext, node: OperatorNode) -> str:
        """Render a _QueryContext into a flat SQL string."""
        # Determine SELECT columns
        if node.output_columns:
            cols = [self._resolve_var(c, ctx.var_mapping) for c in node.output_columns]
        else:
            cols = ["*"]
        select_str = ", ".join(cols)

        parts = [f"SELECT {select_str}"]
        parts.append(f"  FROM {ctx.from_clause}")
        for jc in ctx.join_clauses:
            parts.append(f"  {jc}")
        if ctx.where_conditions:
            parts.append(f"  WHERE {' AND '.join(ctx.where_conditions)}")
        return "\n".join(parts)

    # --- Projection ---

    def _convert_projection(self, node: OperatorNode, params: ProjectionParams) -> str:
        """Projection -> SELECT specific columns FROM (flattened inner)."""
        inputs = node.inputs
        if not inputs:
            raise _ConversionError("Projection requires exactly 1 input")
        if not params.columns:
            raise _ConversionError("Projection requires at least one column")

        # Validate columns exist in input
        available = self._get_node_columns(inputs[0])
        if available:
            for col in params.columns:
                underlying = self._extract_underlying_column(col)
                if underlying not in available:
                    raise _ConversionError(
                        f"Projection column '{col}' (underlying: '{underlying}') "
                        f"not found in input columns: {available}"
                    )

        # Build context from inner node
        ctx = self._build_context(inputs[0])
        col_strs = [self._project_col_to_sql(c, ctx.var_mapping) for c in params.columns]
        select_str = ", ".join(col_strs)

        parts = [f"SELECT {select_str}"]
        parts.append(f"  FROM {ctx.from_clause}")
        for jc in ctx.join_clauses:
            parts.append(f"  {jc}")
        if ctx.where_conditions:
            parts.append(f"  WHERE {' AND '.join(ctx.where_conditions)}")
        return "\n".join(parts)

    def _project_col_to_sql(self, col: str, var_map: dict[str, str]) -> str:
        """Render a projection column, resolving plain names via the var map.

        Aggregates pass through `_col_to_sql` unchanged. For plain column
        names we always emit the *qualified* ``alias.column`` form, with
        ``AS name`` appended when the requested name differs from the
        underlying column (as happens with the ``_1`` / ``_2`` / ``_r2``
        suffixes the operator layer introduces to disambiguate self-joins
        and natural-join overlaps).

        Always qualifying is necessary because the inner relation may join
        multiple tables that share a column name — an unqualified
        reference would be ambiguous and rejected by the database. A
        qualified reference is always unambiguous, and the optional
        ``AS name`` keeps the column visible to outer scopes under the
        name they expect.
        """
        col = col.strip()

        # Aggregate forms — pass through.
        if col.startswith("("):
            return self._col_to_sql(col, var_map)
        for func in _AGGREGATE_FUNCS:
            if col.startswith(f"{func}(") and col.endswith(")"):
                return self._col_to_sql(col, var_map)
        parts = col.split()
        if len(parts) == 2 and parts[0] in _AGGREGATE_FUNCS:
            return self._col_to_sql(col, var_map)

        resolved = self._resolve_var(col, var_map)
        if "." in resolved:
            base = resolved.split(".", 1)[1]
            if base != col:
                return f"{resolved} AS {col}"
            return resolved
        return resolved

    # --- Union ---

    def _convert_union(self, node: OperatorNode) -> str:
        """Union -> (...) UNION (...)."""
        inputs = node.inputs
        if len(inputs) < 2:
            raise _ConversionError("Union requires exactly 2 inputs")
        left_sql = self._convert_node(inputs[0])
        right_sql = self._convert_node(inputs[1])
        left_indented = self._indent_sql(left_sql, 2)
        right_indented = self._indent_sql(right_sql, 2)
        return f"(\n{left_indented}\n)\nUNION\n(\n{right_indented}\n)"

    # --- Division ---

    def _convert_division(self, node: OperatorNode) -> str:
        """Division -> double NOT EXISTS pattern."""
        inputs = node.inputs
        if len(inputs) < 2:
            raise _ConversionError("Division requires exactly 2 inputs")

        left_cols = self._get_node_columns(inputs[0])
        right_cols = self._get_node_columns(inputs[1])
        if not right_cols:
            raise _ConversionError("Division: right input has no columns")

        result_cols = [c for c in left_cols if c not in right_cols]
        if not result_cols:
            raise _ConversionError("Division: no result columns")

        left_table = self._get_table_name(inputs[0])
        right_table = self._get_table_name(inputs[1])

        result_cols_sql = ", ".join(result_cols)
        inner_conds: list[str] = []
        for col in result_cols:
            inner_conds.append(f"t3.{col} = t1.{col}")
        for col in right_cols:
            inner_conds.append(f"t3.{col} = t2.{col}")
        inner_where = " AND ".join(inner_conds)

        return (
            f"SELECT DISTINCT {result_cols_sql} FROM {left_table} t1 "
            f"WHERE NOT EXISTS ("
            f"SELECT * FROM {right_table} t2 "
            f"WHERE NOT EXISTS ("
            f"SELECT * FROM {left_table} t3 "
            f"WHERE {inner_where}"
            f"))"
        )

    # --- Result variables (aggregates + GROUP BY) ---

    def _generate_with_result_vars(
        self,
        tree: OperationTree,
        result_variables: list,
        distinct: bool = False,
        order_by: list | None = None,
        limit: int | None = None,
    ) -> SQLResult:
        """Generate SQL when result_variables are provided (may include aggregates).

        The ``result_variables`` come from the *target* DRC expression, not
        from the operation tree. The planner is free to introduce suffixes
        (``_1``, ``_2``, ``_r2``) onto the tree root's output columns
        during operator application, so the names rarely match the
        target's. They do, however, agree positionally: the equivalence
        checker enforces equal arity and the operators preserve column
        order. We therefore look each target result variable up *by
        position* in the root's ``output_columns``, resolve that name in
        the inner relation context, and emit ``AS target_name`` when the
        underlying name differs from the target name.

        ``distinct`` controls whether the outer SELECT uses ``DISTINCT``.
        It defaults to ``False`` — duplicate elimination should be an
        explicit decision driven by user intent (typically determined by
        a separate analysis of the natural-language question), not an
        unconditional artifact of mapping DRC's set semantics onto SQL's
        bag semantics. When aggregates are present the flag is ignored:
        ``GROUP BY`` (mixed columns and aggregates) or single-row
        aggregation already produce a distinct result.
        """
        has_aggregates = any(isinstance(rv, AggregateVariable) for rv in result_variables)

        ctx = self._build_context(tree.root)
        root_columns = self._get_node_columns(tree.root)

        def _lookup_at(idx: int, fallback_name: str) -> str:
            """Resolve the ``idx``-th tree column, falling back to the
            target name only if the tree lacks output columns."""
            if 0 <= idx < len(root_columns):
                return self._resolve_var(root_columns[idx], ctx.var_mapping)
            return self._resolve_var(fallback_name, ctx.var_mapping)

        select_parts: list[str] = []
        for i, rv in enumerate(result_variables):
            if isinstance(rv, ColumnVariable):
                qualified = _lookup_at(i, rv.name)
                base = qualified.split(".", 1)[-1] if "." in qualified else qualified
                if base != rv.name:
                    select_parts.append(f"{qualified} AS {rv.name}")
                else:
                    select_parts.append(qualified)
            elif isinstance(rv, AggregateVariable):
                qualified_col = _lookup_at(i, rv.column)
                select_parts.append(f"{rv.function}({qualified_col})")

        select_str = ", ".join(select_parts)

        # DRC ``{x1, ..., xn | C}`` is set semantics, but blindly emitting
        # ``DISTINCT`` to match it changes COUNT-over-bag-cardinality
        # results and obscures otherwise-suspicious duplication. We only
        # emit ``DISTINCT`` when the caller explicitly asked for it (via
        # the ``distinct`` flag, typically derived from a question-level
        # analysis of user intent). When aggregates are present the flag
        # is ignored: ``GROUP BY`` or the single-row aggregate result
        # already make the projection distinct.
        if has_aggregates or not distinct:
            select_keyword = "SELECT"
        else:
            select_keyword = "SELECT DISTINCT"
        parts = [f"{select_keyword} {select_str}"]
        parts.append(f"  FROM {ctx.from_clause}")
        for jc in ctx.join_clauses:
            parts.append(f"  {jc}")
        if ctx.where_conditions:
            parts.append(f"  WHERE {' AND '.join(ctx.where_conditions)}")

        if has_aggregates:
            plain_cols: list[str] = []
            for i, rv in enumerate(result_variables):
                if isinstance(rv, ColumnVariable):
                    plain_cols.append(_lookup_at(i, rv.name))
            if plain_cols:
                parts.append(f"  GROUP BY {', '.join(plain_cols)}")

        # ORDER BY / LIMIT layers are non-relational — they live outside
        # the set comprehension. Each ORDER BY key references the target
        # DRC's result-variable namespace, so we resolve via the same
        # positional mapping ``_lookup_at`` uses for SELECT columns.
        if order_by:
            order_parts: list[str] = []
            # Build a name-to-position map so we can find the column
            # underlying a referenced result-variable name.
            rv_name_to_idx: dict[str, int] = {}
            for i, rv in enumerate(result_variables):
                if isinstance(rv, ColumnVariable):
                    rv_name_to_idx[rv.name] = i
                elif isinstance(rv, AggregateVariable):
                    rv_name_to_idx[rv.column] = i

            for crit in order_by:
                col = crit.column
                if not col:
                    return SQLFailure(error="ORDER BY criterion has empty column")
                if col in rv_name_to_idx:
                    qualified = _lookup_at(rv_name_to_idx[col], col)
                else:
                    # Not a result variable — fall back to direct
                    # resolution against the inner var_map. This handles
                    # ORDER BY on a column the user named outside the
                    # SELECT list (rare but allowed in standard SQL).
                    qualified = self._resolve_var(col, ctx.var_mapping)
                if crit.aggregate:
                    qualified = f"{crit.aggregate}({qualified})"
                direction = (crit.direction or "asc").upper()
                if direction not in ("ASC", "DESC"):
                    return SQLFailure(error=f"Invalid ORDER BY direction: {crit.direction}")
                order_parts.append(f"{qualified} {direction}")
            parts.append(f"  ORDER BY {', '.join(order_parts)}")

        if limit is not None:
            if not isinstance(limit, int) or limit <= 0:
                return SQLFailure(error=f"LIMIT must be a positive integer, got {limit!r}")
            parts.append(f"  LIMIT {limit}")

        return SQLSuccess(sql="\n".join(parts))

    # --- Variable resolution ---

    def _resolve_var(self, name: str, var_map: dict[str, str]) -> str:
        """Resolve a DRC variable name to alias.column using the var_map.

        Resolution order:
        1. Already qualified (contains a dot) — return as-is.
        2. Direct hit in var_map — return that.
        3. Bare name (no suffix) and base is in var_map — return that.

        We deliberately do NOT silently fall back from a suffixed name to
        the base name, because that loses the disambiguation a join or
        cartesian product introduced (e.g. ``review_id_1`` vs
        ``review_id_2`` would otherwise both resolve to whichever side
        happens to map ``review_id``).
        """
        # Already qualified
        if "." in name:
            return name
        # Direct lookup
        if name in var_map:
            return var_map[name]
        # No suffix: fall back to the bare name (covers cases where outer
        # references weren't suffixed in the AST — e.g. selection conditions
        # using the original column names).
        base, suffix = _strip_suffix(name)
        if not suffix and base in var_map:
            return var_map[base]
        # Truly unresolved — fail loudly instead of silently emitting a
        # made-up alias.column reference.
        raise _ConversionError(
            f"Cannot resolve variable '{name}' against the current relation. "
            f"Available: {sorted(var_map)}"
        )

    # --- Condition to SQL ---

    def _condition_to_sql(self, condition: DRCCondition, var_map: dict[str, str]) -> str:
        """Convert a DRC condition to SQL with alias qualification."""
        if condition is None:
            raise _ConversionError("Condition is None")

        if isinstance(condition, ComparisonNode):
            return self._comparison_to_sql(condition, var_map)
        elif isinstance(condition, LogicalConnectiveNode):
            return self._logical_to_sql(condition, var_map)
        elif isinstance(condition, NotNode):
            operand = self._condition_to_sql(condition.operand, var_map)
            if not operand:
                return ""
            return f"NOT ({operand})"
        elif isinstance(condition, VariableRefNode):
            return self._resolve_var(condition.name, var_map)
        elif isinstance(condition, LiteralNode):
            return self._literal_to_sql(condition)
        elif isinstance(condition, MembershipNode):
            return ""  # Structural, not a filter
        elif isinstance(condition, QuantifierNode):
            return self._quantifier_to_sql(condition, var_map)
        elif isinstance(condition, FunctionCallNode):
            return self._function_to_sql(condition, var_map)
        else:
            raise _ConversionError(f"Unsupported condition type: {type(condition).__name__}")

    def _comparison_to_sql(self, node: ComparisonNode, var_map: dict[str, str]) -> str:
        """Convert a comparison node to SQL with normalization."""
        left_sql = self._condition_to_sql(node.left, var_map)
        right_sql = self._condition_to_sql(node.right, var_map)

        # Normalize: put simple column on left when right is complex expr
        if self._is_complex_expr(node.left) and isinstance(node.right, VariableRefNode):
            left_sql, right_sql = right_sql, left_sql
            op = _flip_operator(node.operator)
        else:
            op = node.operator

        return f"{left_sql} {op} {right_sql}"

    def _logical_to_sql(self, node: LogicalConnectiveNode, var_map: dict[str, str]) -> str:
        """Convert a logical connective to SQL."""
        left = self._condition_to_sql(node.left, var_map)
        right = self._condition_to_sql(node.right, var_map)
        op = node.operator.upper()
        if op == "IMPLIES":
            return f"(NOT ({left}) OR {right})"
        parts = [p for p in [left, right] if p]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return f"({parts[0]} {op} {parts[1]})"

    def _literal_to_sql(self, node: LiteralNode) -> str:
        """Convert a literal to SQL."""
        if node.data_type == "string":
            val = str(node.value)
            escaped = val.replace("'", "''")
            if _is_date_string(val):
                return f"DATE '{escaped}'"
            return f"'{escaped}'"
        return str(node.value)

    def _function_to_sql(self, node: FunctionCallNode, var_map: dict[str, str]) -> str:
        """Convert a function call to SQL."""
        if node.function == "CURRENT_DATE":
            return "CURRENT_DATE"
        elif node.function == "DATE_SUB" and len(node.arguments) == 2:
            base = self._condition_to_sql(node.arguments[0], var_map)
            days_str = self._condition_to_sql(node.arguments[1], var_map)
            interval = _days_to_interval(days_str)
            return f"{base} - INTERVAL '{interval}'"
        elif node.function == "DATE_ADD" and len(node.arguments) == 2:
            base = self._condition_to_sql(node.arguments[0], var_map)
            days_str = self._condition_to_sql(node.arguments[1], var_map)
            interval = _days_to_interval(days_str)
            return f"{base} + INTERVAL '{interval}'"
        elif node.function == "DATEDIFF" and len(node.arguments) == 2:
            left = self._condition_to_sql(node.arguments[0], var_map)
            right = self._condition_to_sql(node.arguments[1], var_map)
            return f"({left} - {right})"
        else:
            if not node.arguments:
                return node.function
            args = ", ".join(self._condition_to_sql(a, var_map) for a in node.arguments)
            return f"{node.function}({args})"

    def _quantifier_to_sql(self, node: QuantifierNode, outer_var_map: dict[str, str]) -> str:
        """Convert a quantifier (EXISTS/FORALL) to a correlated subquery."""
        # Simple case: body is a MembershipNode
        if isinstance(node.body, MembershipNode):
            return self._quantifier_membership(node, node.body, outer_var_map)

        # Body is AND with membership inside
        if isinstance(node.body, LogicalConnectiveNode) and node.body.operator == "and":
            table_name = self._extract_table_from_condition(node.body)
            if table_name:
                return self._quantifier_and_membership(node, table_name, outer_var_map)

        # Fallback: generic body
        body_sql = self._condition_to_sql(node.body, outer_var_map)
        if not body_sql:
            return ""
        if node.kind == "exists":
            return f"EXISTS (SELECT 1 WHERE {body_sql})"
        elif node.kind == "forall":
            return f"NOT EXISTS (SELECT 1 WHERE NOT ({body_sql}))"
        return body_sql

    def _quantifier_membership(self, node: QuantifierNode, membership: MembershipNode, outer_var_map: dict[str, str]) -> str:
        """Handle quantifier with a direct MembershipNode body."""
        table = membership.relation
        inner_alias = self._aliases.next_alias(table)
        quantified = set(node.variables)
        correlated = [v for v in membership.variables if v not in quantified]

        where_parts: list[str] = []
        for v in correlated:
            outer_ref = outer_var_map.get(v, v)
            where_parts.append(f"{inner_alias}.{v} = {outer_ref}")
        where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""

        if node.kind == "exists":
            return f"EXISTS (SELECT 1 FROM {table} {inner_alias}{where_clause})"
        elif node.kind == "forall":
            return f"NOT EXISTS (SELECT 1 FROM {table} {inner_alias}{where_clause})"
        return ""

    def _quantifier_and_membership(self, node: QuantifierNode, table_name: str, outer_var_map: dict[str, str]) -> str:
        """Handle quantifier with AND body containing a membership."""
        inner_alias = self._aliases.next_alias(table_name)
        membership = self._find_membership(node.body)
        quantified = set(node.variables)
        correlated: list[str] = []
        if membership:
            correlated = [v for v in membership.variables if v not in quantified]

        # Build inner var_map for the subquery
        inner_var_map = dict(outer_var_map)
        if membership:
            for v in membership.variables:
                inner_var_map[v] = f"{inner_alias}.{v}"

        body_sql = self._condition_to_sql_skip_membership(node.body, inner_var_map)

        where_parts: list[str] = []
        for v in correlated:
            outer_ref = outer_var_map.get(v, v)
            where_parts.append(f"{inner_alias}.{v} = {outer_ref}")
        if body_sql:
            where_parts.append(body_sql)
        where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""

        if node.kind == "exists":
            return f"EXISTS (SELECT 1 FROM {table_name} {inner_alias}{where_clause})"
        elif node.kind == "forall":
            return f"NOT EXISTS (SELECT 1 FROM {table_name} {inner_alias}{where_clause})"
        return ""

    def _condition_to_sql_skip_membership(self, condition: DRCCondition, var_map: dict[str, str]) -> str:
        """Convert condition to SQL, skipping MembershipNodes."""
        if isinstance(condition, LogicalConnectiveNode) and condition.operator == "and":
            left = self._condition_to_sql_skip_membership(condition.left, var_map)
            right = self._condition_to_sql_skip_membership(condition.right, var_map)
            parts = [p for p in [left, right] if p]
            if not parts:
                return ""
            return " AND ".join(parts)
        if isinstance(condition, MembershipNode):
            return ""
        return self._condition_to_sql(condition, var_map)

    # --- Utility methods ---

    def _extract_table_from_condition(self, node) -> str | None:
        """Extract a table name from a MembershipNode within a condition tree."""
        if isinstance(node, MembershipNode):
            return node.relation
        if isinstance(node, LogicalConnectiveNode):
            left = self._extract_table_from_condition(node.left)
            if left:
                return left
            return self._extract_table_from_condition(node.right)
        return None

    def _find_membership(self, node):
        """Find the first MembershipNode in a condition tree."""
        if isinstance(node, MembershipNode):
            return node
        if isinstance(node, LogicalConnectiveNode):
            left = self._find_membership(node.left)
            if left:
                return left
            return self._find_membership(node.right)
        return None

    def _is_complex_expr(self, node) -> bool:
        """Check if a node is a complex expression (not a simple variable ref)."""
        return isinstance(node, (FunctionCallNode, LiteralNode))

    def _col_to_sql(self, col: str, var_map: dict[str, str]) -> str:
        """Convert a column spec to SQL syntax (handles aggregates)."""
        col = col.strip()
        # Handle "(FUNC col)" format
        if col.startswith("(") and col.endswith(")"):
            inner = col[1:-1].strip()
            parts = inner.split()
            if len(parts) == 2 and parts[0] in _AGGREGATE_FUNCS:
                return f"{parts[0]}({parts[1]})"
        # Handle "FUNC(col)" format
        for func in _AGGREGATE_FUNCS:
            if col.startswith(f"{func}(") and col.endswith(")"):
                return col
        # Handle "FUNC col" format
        parts = col.split()
        if len(parts) == 2 and parts[0] in _AGGREGATE_FUNCS:
            return f"{parts[0]}({parts[1]})"
        return col

    def _extract_underlying_column(self, col: str) -> str:
        """Extract the underlying column name from a column spec."""
        col = col.strip()
        if col.startswith("(") and col.endswith(")"):
            inner = col[1:-1].strip()
            parts = inner.split()
            if len(parts) == 2 and parts[0] in _AGGREGATE_FUNCS:
                return parts[1]
        for func in _AGGREGATE_FUNCS:
            if col.startswith(f"{func}(") and col.endswith(")"):
                return col[len(func) + 1:-1].strip()
        parts = col.split()
        if len(parts) == 2 and parts[0] in _AGGREGATE_FUNCS:
            return parts[1]
        return col

    def _get_node_columns(self, node: OperationNode) -> list[str]:
        """Get the output columns of a node."""
        if isinstance(node, TableLeafNode):
            return node.columns
        elif isinstance(node, OperatorNode):
            return node.output_columns
        return []

    def _get_table_name(self, node: OperationNode) -> str:
        """Get the base table name from a node (for simple cases)."""
        if isinstance(node, TableLeafNode):
            return node.table_name
        sql = self._convert_node(node)
        return f"({sql})"

    @staticmethod
    def _indent_sql(sql: str, indent: int = 4) -> str:
        """Indent each line of a SQL string."""
        pad = " " * indent
        return "\n".join(f"{pad}{line}" for line in sql.split("\n"))


# --- Public API ---


def convert_to_sql(
    tree: OperationTree,
    result_variables: list | None = None,
    distinct: bool = False,
    order_by: list | None = None,
    limit: int | None = None,
) -> SQLResult:
    """Convert an OperationTree into a flat SQL SELECT statement.

    If result_variables contains a mix of plain columns and aggregates,
    emits GROUP BY for the plain columns.

    Args:
        tree: The operation tree.
        result_variables: The target DRC expression's result variables (optional).
        distinct: If True (and no aggregates are present), emit
            ``SELECT DISTINCT`` to enforce set semantics on the outer
            projection. Defaults to False; the caller is expected to
            decide based on user intent (e.g. via a question-level
            analysis from the LLM).
        order_by: Optional list of ``SortCriterion`` from an extended-DRC
            ``ORDER BY`` wrapper. Emits ``ORDER BY`` on the outer SELECT.
        limit: Optional positive integer from an extended-DRC ``LIMIT``
            wrapper. Emits ``LIMIT`` on the outer SELECT.
    """
    gen = _SqlGenerator()
    return gen.generate(
        tree,
        result_variables,
        distinct=distinct,
        order_by=order_by,
        limit=limit,
    )


def convert_query_to_sql(
    tree: OperationTree,
    query,
    distinct: bool = False,
) -> SQLResult:
    """Convert a tree + query (DRC, optionally wrapped in order-by/limit)
    into a flat SQL SELECT statement.

    The query's inner DRC supplies the result-variable list; the
    ``ORDER BY`` / ``LIMIT`` layers (if any) become outer-SELECT clauses.
    """
    from text_to_sql_planner.types.drc import (
        DRCExpression,
        LimitExpression,
        OrderByExpression,
        query_inner_drc,
        query_limit,
        query_order_by,
    )

    if query is None:
        return SQLFailure(error="query is None")

    inner = query_inner_drc(query)
    if not isinstance(inner, DRCExpression):
        return SQLFailure(error=f"Unexpected query inner type: {type(inner).__name__}")

    ob = query_order_by(query)
    lim = query_limit(query)

    return convert_to_sql(
        tree,
        result_variables=inner.result_variables,
        distinct=distinct,
        order_by=ob.criteria if ob is not None else None,
        limit=lim.n if lim is not None else None,
    )
