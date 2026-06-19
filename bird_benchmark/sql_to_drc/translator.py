"""Schema-aware translator: SelectStatement → DRC.

This module is the second stage of the SQL→DRC pipeline. It consumes the
:class:`~bird_benchmark.sql_to_drc.ast.SelectStatement` produced by
:mod:`bird_benchmark.sql_to_drc.parser` and produces a
:class:`text_to_sql_planner.types.drc.QueryExpression` (a
``DRCExpression`` optionally wrapped in ``OrderByExpression`` /
``LimitExpression``).

Translation strategy
--------------------

For each SELECT level the translator builds a :class:`Scope` carrying:

- ``tables``      – ``alias -> table_name`` for every table introduced at
  this level. Both the alias (when present) and the bare table name are
  registered so ``a.col`` and ``my_table.col`` both resolve.
- ``bound_vars``  – ``(alias, column_name) -> fresh DRC variable name``
  bindings introduced by the FROM clause's memberships.
- ``parent``      – pointer to the enclosing scope, walked for correlated
  subqueries.

Correlated outer column references resolve by walking ``parent`` until a
matching binding is found. The inner scope reuses the *parent's* variable
name in a ``VariableRefNode`` rather than re-binding (per design).

For each SELECT the translator emits the following DRC shape::

    LIMIT? · ORDER_BY? · DRCExpression(
        result_variables = [...],
        condition = AND(memberships, WHERE-tree),
    )

Subqueries (``EXISTS`` and ``IN (SELECT ...)``) are translated as
``QuantifierNode(kind="exists", variables=<inner bound vars>,
body=<inner conjunction>)``. ``IN (SELECT s FROM ...)`` adds an extra
equality between the outer LHS and the inner result variable to the body.

The translator never raises for malformed input. It always returns
either a ``QueryExpression`` or a structured ``ConverterError`` so the
runner can record a failed Test_Case and continue. Internally it uses a
private ``_TranslateFail`` exception for early-out plumbing; that
exception is caught at the public entry point and converted into a
``ConverterError`` return value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from text_to_sql_planner.types.drc import (
    AggregateVariable,
    ArithmeticNode,
    ColumnVariable,
    ComparisonNode,
    DRCCondition,
    DRCExpression,
    FunctionCallNode,
    LimitExpression,
    LiteralNode,
    LogicalConnectiveNode,
    MembershipNode,
    NotNode,
    OrderByExpression,
    QuantifierNode,
    QueryExpression,
    ResultVariable,
    SortCriterion,
    VariableRefNode,
)

from . import ast as sql_ast
from .schema import ColumnType, parse_schema
from ..types import ConverterError


# --- Internal control flow -------------------------------------------------


class _TranslateFail(Exception):
    """Carrier for ``ConverterError`` so recursive helpers can bail out.

    Caught once at the public :func:`translate` entry point. Callers of
    the converter never see this exception.
    """

    def __init__(self, error: ConverterError) -> None:
        self.error = error
        super().__init__(error.message)


# --- Scope -----------------------------------------------------------------


@dataclass
class Scope:
    """One translation scope, one per SELECT level.

    See the design's "Schema-aware translator" section.
    """

    # alias-or-name -> table name (lowercased keys)
    tables: dict[str, str] = field(default_factory=dict)
    # (table_alias_lower, column_name_lower) -> fresh DRC variable name
    bound_vars: dict[tuple[str, str], str] = field(default_factory=dict)
    # The order columns were bound for each table alias. Used by ``*``
    # expansion in the SELECT list and by aggregate ``COUNT(*)``.
    columns_by_alias: dict[str, list[str]] = field(default_factory=dict)
    parent: Optional["Scope"] = None

    def lookup_var(
        self, alias: str | None, column: str
    ) -> tuple[str, "Scope"] | None:
        """Walk parents to resolve ``(alias, column)`` to a bound variable.

        Returns ``(variable_name, owning_scope)`` on success or ``None``
        when no binding is found in this scope or any parent. The owning
        scope is returned so callers can tell whether the binding came
        from the current scope or an outer one — used by translation of
        correlated subqueries.
        """

        column_l = column.lower()
        cur: Optional[Scope] = self
        while cur is not None:
            if alias is not None:
                key = (alias.lower(), column_l)
                if key in cur.bound_vars:
                    return cur.bound_vars[key], cur
            else:
                # Unqualified column: succeed only if exactly one binding
                # in this scope matches; otherwise walk to the parent.
                matches = [
                    (k, v)
                    for k, v in cur.bound_vars.items()
                    if k[1] == column_l
                ]
                if len(matches) == 1:
                    return matches[0][1], cur
                if len(matches) > 1:
                    # Ambiguous within a scope: do not silently pick one;
                    # report a structured error from the call site.
                    return None
            cur = cur.parent
        return None


# --- Helpers ---------------------------------------------------------------


def _conjoin(parts: Iterable[DRCCondition | None]) -> DRCCondition:
    """Combine a list of conditions with ``and``.

    Empty or all-``None`` input yields a trivially-true literal so callers
    don't have to special-case the empty case. Single-element input is
    returned as-is so we don't introduce gratuitous ``and`` nodes.
    """

    real = [p for p in parts if p is not None]
    if not real:
        return LiteralNode(value=1, data_type="number")
    if len(real) == 1:
        return real[0]
    result: DRCCondition = real[0]
    for nxt in real[1:]:
        result = LogicalConnectiveNode(
            operator="and", left=result, right=nxt
        )
    return result


def _disjoin(parts: list[DRCCondition]) -> DRCCondition:
    """Combine a non-empty list of conditions with ``or``."""

    if not parts:
        # An ``IN ()`` (empty list) is a contradiction; emit ``0 = 1``.
        return ComparisonNode(
            operator="=",
            left=LiteralNode(value=0, data_type="number"),
            right=LiteralNode(value=1, data_type="number"),
        )
    if len(parts) == 1:
        return parts[0]
    result: DRCCondition = parts[0]
    for nxt in parts[1:]:
        result = LogicalConnectiveNode(operator="or", left=result, right=nxt)
    return result


def _literal_node_from_ast(lit: sql_ast.Literal) -> LiteralNode:
    """Convert a parser ``Literal`` to a DRC ``LiteralNode``.

    DRC's ``LiteralNode`` only distinguishes ``"string"`` from
    ``"number"`` (Int and Real share the latter), which matches the
    SMT-LIB type lattice cvc5 consumes.
    """

    if lit.data_type == "string":
        return LiteralNode(value=lit.value, data_type="string")
    return LiteralNode(value=lit.value, data_type="number")


def _columns_for_table_from_schema(
    schema_map: dict[tuple[str, str], ColumnType], table: str
) -> list[str]:
    """Return the column names of ``table`` in CREATE-TABLE source order.

    The schema parser populates the dict by inserting each column in the
    order it appears in the source, so iterating ``schema_map.items()``
    yields the canonical column order for each table. Empty list is
    returned when the table is not present.
    """

    table_l = table.lower()
    out: list[str] = []
    for (t, col), _typ in schema_map.items():
        if t == table_l:
            out.append(col)
    return out


def _collect_referenced_columns(
    stmt: sql_ast.SelectStatement, table_alias: str
) -> list[str]:
    """Fallback: collect column names referenced under ``table_alias``.

    Used when the active schema does not contain a table referenced in
    FROM. Walks the SelectStatement looking for ``ColumnRef`` nodes that
    target the given alias (or are unqualified when there is only one
    table). Preserves first-seen order and de-duplicates.
    """

    seen: dict[str, None] = {}
    target = table_alias.lower()

    def visit_expr(expr: sql_ast.Expression) -> None:
        if isinstance(expr, sql_ast.ColumnRef):
            if expr.qualifier is None or expr.qualifier.lower() == target:
                seen.setdefault(expr.name.lower(), None)
            return
        if isinstance(expr, sql_ast.BinaryOp):
            visit_expr(expr.left)
            visit_expr(expr.right)
            return
        if isinstance(expr, sql_ast.UnaryOp):
            visit_expr(expr.operand)
            return
        if isinstance(expr, sql_ast.FunctionCall):
            for arg in expr.args:
                visit_expr(arg)
            return
        if isinstance(expr, sql_ast.Aggregate):
            if isinstance(expr.column, sql_ast.ColumnRef):
                visit_expr(expr.column)
            return
        if isinstance(expr, sql_ast.InList):
            visit_expr(expr.left)
            for v in expr.values:
                visit_expr(v)
            return
        if isinstance(expr, sql_ast.InSubquery):
            visit_expr(expr.left)
            return
        # ExistsExpr / Literal: no referenced columns at this level.

    for item in stmt.select_list:
        visit_expr(item.expr)
    if stmt.where is not None:
        visit_expr(stmt.where)
    for c in stmt.group_by:
        visit_expr(c)
    for k in stmt.order_by:
        visit_expr(k.expr)
    return list(seen.keys())


# --- Translator ------------------------------------------------------------


class _Translator:
    """Walks the SelectStatement AST and emits DRC.

    One translator instance is reused across all nested SELECTs inside a
    single conversion. The instance owns a counter that produces fresh
    variable names so names are unique across every scope, which keeps
    correlated subqueries from accidentally rebinding outer variables.
    """

    def __init__(self, schema_map: dict[tuple[str, str], ColumnType]) -> None:
        self.schema_map = schema_map
        self._counter = 0

    # ----- fresh-name helpers --------------------------------------------

    def _fresh(self, table: str, column: str) -> str:
        """Produce a new DRC variable name unique within this conversion."""
        self._counter += 1
        # Sanitise so the resulting identifier is always a plain
        # ``[A-Za-z0-9_]+`` token. SMT-LIB and the planner's pretty
        # printer both expect plain identifiers; quoted column names
        # would otherwise leak through.
        safe_table = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in table)
        safe_col = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in column)
        return f"v_{safe_table}_{safe_col}_{self._counter}"

    # ----- error helpers --------------------------------------------------

    @staticmethod
    def _fail_unsupported(feature: str, message: str, pos: sql_ast.Position) -> _TranslateFail:
        return _TranslateFail(
            ConverterError(
                kind="unsupported_feature",
                message=message,
                feature=feature,
                line=pos.line,
                column=pos.column,
            )
        )

    @staticmethod
    def _fail_unbound(message: str, pos: sql_ast.Position) -> _TranslateFail:
        return _TranslateFail(
            ConverterError(
                kind="unbound_reference",
                message=message,
                line=pos.line,
                column=pos.column,
            )
        )

    # ----- SelectStatement ------------------------------------------------

    def translate_select(
        self,
        stmt: sql_ast.SelectStatement,
        parent: Optional[Scope],
    ) -> tuple[QueryExpression, Scope]:
        """Translate one SELECT level.

        Returns the query expression (possibly wrapped in ORDER BY /
        LIMIT) plus the scope that the SELECT bound — the caller uses
        this scope for correlated-subquery references that originate in
        an enclosing expression.
        """

        scope = Scope(parent=parent)

        # ---- FROM: build memberships and populate scope ---------------
        memberships = self._translate_from(stmt, scope)

        # ---- WHERE -----------------------------------------------------
        where_cond: DRCCondition | None = None
        if stmt.where is not None:
            where_cond = self._translate_expr(stmt.where, scope)

        # ---- HAVING ----------------------------------------------------
        having_cond: DRCCondition | None = None
        if stmt.having is not None:
            having_cond = self._translate_expr(stmt.having, scope)

        body = _conjoin([*memberships, where_cond, having_cond])

        # ---- result_variables -----------------------------------------
        result_variables = self._translate_select_list(stmt, scope)

        # ---- Existential closure --------------------------------------
        # Every name introduced by FROM-clause memberships that is NOT a
        # result variable must be existentially quantified. The DRC
        # contract is "no free variables in the condition other than
        # the result variables" — the planner-side LLM follows this
        # rule explicitly, but the SQL→DRC translator used to leave
        # filter columns as free constants. That broke equivalence
        # checks against the planner: cvc5 would see one side bind
        # ``City`` with ``∃City. … ∧ City="Adelanto"`` and the other
        # side leave ``v_city_10`` as a free String constant. The two
        # are not logically the same shape — cvc5 picks ``v_city_10
        # ≠ "Adelanto"`` as a counterexample and reports
        # ``not_equivalent`` even though the original SQL queries
        # describe identical relations (dev_78 / dev_1425 / dev_757 /
        # dev_1375 in run-10).
        result_var_names: set[str] = set()
        for rv in result_variables:
            if isinstance(rv, ColumnVariable):
                result_var_names.add(rv.name)
            elif isinstance(rv, AggregateVariable):
                # The aggregated column is a free variable on the
                # outside (the aggregate replaces it positionally), so
                # NOT existentially bound; same for ``COUNT(*)`` whose
                # column is the literal ``"*"``.
                result_var_names.add(rv.column)
        # Names introduced by FROM-clause bindings that are not in the
        # SELECT list become existentially-quantified bound names.
        bound_names = list(scope.bound_vars.values())
        # Preserve introduction order while deduplicating.
        seen: set[str] = set()
        free_filter_names: list[str] = []
        for name in bound_names:
            if name in result_var_names or name in seen:
                continue
            seen.add(name)
            free_filter_names.append(name)
        if free_filter_names:
            body = QuantifierNode(
                kind="exists",
                variables=free_filter_names,
                body=body,
            )

        drc_expr = DRCExpression(
            result_variables=result_variables,
            condition=body,
        )

        # ---- ORDER BY / LIMIT wrappers ---------------------------------
        wrapped: QueryExpression = drc_expr
        if stmt.order_by:
            criteria = [
                self._translate_order_key(k, scope) for k in stmt.order_by
            ]
            wrapped = OrderByExpression(criteria=criteria, inner=drc_expr)
        if stmt.limit is not None:
            wrapped = LimitExpression(n=stmt.limit, inner=wrapped)

        return wrapped, scope

    # ----- FROM / JOIN ---------------------------------------------------

    def _translate_from(
        self, stmt: sql_ast.SelectStatement, scope: Scope
    ) -> list[DRCCondition]:
        """Bind every FROM/JOIN table in the scope and emit memberships.

        Returns the list of ``MembershipNode`` plus join-ON conditions in
        the order they appear in the source.
        """

        memberships: list[DRCCondition] = []
        source = stmt.from_source
        if isinstance(source, sql_ast.TableRef):
            memberships.append(self._bind_table(source, scope, stmt))
            return memberships
        if isinstance(source, sql_ast.DerivedTable):
            memberships.extend(self._bind_derived_table(source, scope))
            return memberships

        # JoinChain: base table plus a list of INNER JOINs.
        if isinstance(source.base, sql_ast.DerivedTable):
            memberships.extend(self._bind_derived_table(source.base, scope))
        else:
            memberships.append(self._bind_table(source.base, scope, stmt))
        for join in source.joins:
            if isinstance(join.right, sql_ast.DerivedTable):
                memberships.extend(self._bind_derived_table(join.right, scope))
            else:
                memberships.append(self._bind_table(join.right, scope, stmt))
            memberships.append(self._translate_expr(join.on, scope))
        return memberships

    def _bind_derived_table(
        self,
        ref: "sql_ast.DerivedTable",
        scope: Scope,
    ) -> list[DRCCondition]:
        """Bind a parenthesised SELECT subquery as a FROM source.

        The subquery is translated recursively. Its body (conditions
        and memberships, ∃-bound where appropriate) becomes part of
        the outer scope. Each of the subquery's projected columns is
        registered as ``(alias, col_name)`` in the outer
        ``scope.bound_vars`` so the outer SELECT / WHERE / ORDER BY
        can reference them as ``alias.col`` or bare ``col``.

        Aliases on the inner ``select_list`` items take precedence
        over the underlying expression name. For aggregates without
        an alias we synthesise a name (``func_<n>``) so the column is
        still addressable. cvc5 sees aggregate-derived columns as
        unconstrained variables — the right model for an opaque
        aggregate result that the outer query treats as a value.

        Returns the list of DRC conditions to conjoin into the outer
        body. The first element is always the inner subquery's body
        (wrapped in an existential over its non-projected bindings);
        subsequent elements are empty in the common case.
        """

        sub_query, sub_scope = self.translate_select(
            ref.subquery, parent=scope,
        )
        sub_drc = _strip_wrappers(sub_query)

        alias_l = ref.alias.lower()
        scope.tables[alias_l] = ref.alias

        # ``column_names_in_outer`` is the ordered list of column
        # names the outer query can reference on this alias. For each
        # inner select-list item we use its alias if present,
        # otherwise the column name (for ColumnRef) or a synthetic
        # name (for Aggregate / computed expressions).
        column_names_in_outer: list[str] = []
        # ``inner_var_names`` is the parallel list of DRC variable
        # names the inner DRC uses for those columns.
        inner_var_names: list[str] = []
        # Aggregate columns get a fresh outer-scope variable (cvc5 sees
        # them as unconstrained) plus a structural placeholder in the
        # inner DRC's body. We track which positions are aggregates so
        # the outer scope can bind them appropriately.
        for i, item in enumerate(ref.subquery.select_list):
            inner_rv = sub_drc.result_variables[i] if i < len(sub_drc.result_variables) else None
            outer_col_name = self._derived_column_name(item, inner_rv, i)
            if outer_col_name is None:
                continue
            column_names_in_outer.append(outer_col_name)
            # For ColumnVariable result-vars, the inner DRC's variable
            # is the bound name we expose. For AggregateVariable
            # result-vars, we mint a fresh name (the inner aggregate
            # has no per-row variable name).
            if isinstance(inner_rv, ColumnVariable):
                inner_var_names.append(inner_rv.name)
            elif isinstance(inner_rv, AggregateVariable):
                fresh = self._fresh(ref.alias, outer_col_name)
                inner_var_names.append(fresh)
            else:
                fresh = self._fresh(ref.alias, outer_col_name)
                inner_var_names.append(fresh)

        # Register the alias.column lookups in the outer scope.
        for col_name, var_name in zip(column_names_in_outer, inner_var_names):
            scope.bound_vars[(alias_l, col_name.lower())] = var_name
        scope.columns_by_alias[alias_l] = list(column_names_in_outer)

        # Names introduced inside the subquery that are NOT the
        # exposed columns become existentially bound. The outer
        # scope shouldn't see them.
        exposed = set(inner_var_names)
        sub_inner_vars = list(sub_scope.bound_vars.values())
        ex_bound: list[str] = []
        seen: set[str] = set()
        for v in sub_inner_vars:
            if v in exposed or v in seen:
                continue
            seen.add(v)
            ex_bound.append(v)

        body = sub_drc.condition
        if ex_bound:
            body = QuantifierNode(
                kind="exists", variables=ex_bound, body=body,
            )
        return [body]

    def _derived_column_name(
        self,
        item: "sql_ast.SelectItem",
        rv,
        position: int,
    ) -> str | None:
        """Pick the outer-scope column name for one inner select-list item."""
        if item.alias:
            return item.alias
        if isinstance(item.expr, sql_ast.ColumnRef):
            return item.expr.name
        if isinstance(rv, AggregateVariable):
            return f"{rv.function.lower()}_{position}"
        if isinstance(rv, ColumnVariable):
            return rv.name
        return None

    def _bind_table(
        self,
        ref: sql_ast.TableRef,
        scope: Scope,
        stmt: sql_ast.SelectStatement,
    ) -> MembershipNode:
        """Bind a single table reference into ``scope`` and emit its membership.

        The alias is the explicit ``AS`` alias when present, otherwise
        the bare table name. Columns come from the schema map (preferred)
        or, when the schema does not know the table, from the column
        references found in the query body. The latter is how tests with
        empty / minimal schemas keep working.
        """

        alias = (ref.alias if ref.alias is not None else ref.name)
        alias_l = alias.lower()
        scope.tables[alias_l] = ref.name
        # Also register the bare table name so ``my_table.col`` still
        # resolves when the user supplied an alias. Aliases shadow the
        # bare name if they happen to collide.
        scope.tables.setdefault(ref.name.lower(), ref.name)

        columns = _columns_for_table_from_schema(self.schema_map, ref.name)
        if not columns:
            # Schema does not know this table; fall back to the columns
            # actually referenced in the SELECT level so we can still
            # emit a well-formed membership.
            columns = _collect_referenced_columns(stmt, alias)
        if not columns:
            # No information at all; emit a single placeholder so the
            # MembershipNode at least carries the relation name through
            # to cvc5. This keeps malformed gold queries from crashing
            # the suite — they simply won't be equivalent to anything.
            columns = ["__row__"]

        var_names: list[str] = []
        for col in columns:
            var = self._fresh(ref.name, col)
            scope.bound_vars[(alias_l, col.lower())] = var
            var_names.append(var)
        scope.columns_by_alias[alias_l] = list(columns)

        return MembershipNode(variables=var_names, relation=ref.name)

    # ----- Expressions ---------------------------------------------------

    def _translate_expr(
        self, expr: sql_ast.Expression, scope: Scope
    ) -> DRCCondition:
        """Translate a parsed Expression into a DRC condition / value node.

        DRC nominally distinguishes "boolean conditions" from "value
        expressions", but the existing ``DRCCondition`` union already
        carries both shapes (``ComparisonNode`` is a boolean,
        ``ArithmeticNode`` / ``LiteralNode`` / ``VariableRefNode`` are
        values, ``FunctionCallNode`` is either). We rely on the planner's
        downstream consumers to interpret the shape from context.
        """

        if isinstance(expr, sql_ast.Literal):
            return self._translate_literal(expr, scope)

        if isinstance(expr, sql_ast.ColumnRef):
            return self._translate_column_ref(expr, scope)

        if isinstance(expr, sql_ast.BinaryOp):
            return self._translate_binary(expr, scope)

        if isinstance(expr, sql_ast.UnaryOp):
            return self._translate_unary(expr, scope)

        if isinstance(expr, sql_ast.FunctionCall):
            return self._translate_function_call(expr, scope)

        if isinstance(expr, sql_ast.Aggregate):
            # Aggregates appear in select_list and order_by, not in WHERE
            # / ON / etc. Reaching this case from inside an expression
            # tree is a parser bug: we still produce a sane shape rather
            # than crash, by converting the aggregate to a function call
            # over a variable reference.
            col = expr.column
            args: list[DRCCondition] = []
            if isinstance(col, sql_ast.ColumnRef):
                args.append(self._translate_column_ref(col, scope))
            return FunctionCallNode(function=expr.function, arguments=args)

        if isinstance(expr, sql_ast.InList):
            return self._translate_in_list(expr, scope)

        if isinstance(expr, sql_ast.InSubquery):
            return self._translate_in_subquery(expr, scope)

        if isinstance(expr, sql_ast.ExistsExpr):
            return self._translate_exists(expr, scope)

        # Reached only if a new AST node is added without updating the
        # translator. Better to fail loudly than to silently emit junk.
        raise self._fail_unsupported(
            "unknown_expression",
            f"unsupported expression node: {type(expr).__name__}",
            sql_ast.Position(line=1, column=1),
        )

    def _translate_literal(
        self, lit: sql_ast.Literal, scope: Scope
    ) -> DRCCondition:
        return _literal_node_from_ast(lit)

    def _translate_column_ref(
        self, ref: sql_ast.ColumnRef, scope: Scope
    ) -> DRCCondition:
        """Resolve a column reference, applying the SQLite identifier fallback.

        Per Req 13.2 / 13.3, a double-quoted token that does not match
        any table or column in the active schema falls back to a string
        literal. Unquoted column references that fail to resolve are
        reported as ``ConverterError(kind="unbound_reference")``.
        """

        # Fast path: try to resolve as an identifier in the scope chain.
        var = scope.lookup_var(ref.qualifier, ref.name)
        if var is not None:
            return VariableRefNode(name=var[0])

        # SQLite identifier-fallback for double-quoted tokens (Req 13.3):
        # if the source token was double-quoted and does not match any
        # name reachable from the current scope, treat it as a string
        # literal whose value is the unquoted body. We only apply this
        # to tokens marked ``quoted`` so that an unquoted typo is still
        # reported as an unbound reference rather than coerced silently.
        if ref.quoted and ref.qualifier is None:
            return LiteralNode(value=ref.name, data_type="string")

        raise self._fail_unbound(
            f"unbound column reference: "
            f"{(ref.qualifier + '.') if ref.qualifier else ''}{ref.name}",
            ref.pos,
        )

    def _translate_binary(
        self, expr: sql_ast.BinaryOp, scope: Scope
    ) -> DRCCondition:
        op = expr.op
        if op in ("AND", "OR"):
            return LogicalConnectiveNode(
                operator="and" if op == "AND" else "or",
                left=self._translate_expr(expr.left, scope),
                right=self._translate_expr(expr.right, scope),
            )
        if op in ("=", "!=", "<", ">", "<=", ">="):
            return ComparisonNode(
                operator=op,  # type: ignore[arg-type]
                left=self._translate_expr(expr.left, scope),
                right=self._translate_expr(expr.right, scope),
            )
        if op == "||":
            # SQLite ``||`` is string concatenation (Req 13.4). The DRC
            # has no dedicated concat node; we model it as a function
            # call so cvc5 sees a single nominal symbol to reason about.
            return FunctionCallNode(
                function="concat",
                arguments=[
                    self._translate_expr(expr.left, scope),
                    self._translate_expr(expr.right, scope),
                ],
            )
        if op == "LIKE":
            # ``lhs LIKE pattern`` — there's no built-in LIKE in DRC, so
            # we model it as an uninterpreted Bool predicate
            # ``(LIKE lhs pattern)``. The SMT converter declares
            # ``LIKE`` once (with the inferred operand sorts) and both
            # sides of an equivalence check share the same predicate
            # symbol, so a generated ``LIKE Title "%data%"`` and the
            # gold's identical call are recognised as equivalent
            # without cvc5 needing to interpret SQL pattern semantics.
            return FunctionCallNode(
                function="LIKE",
                arguments=[
                    self._translate_expr(expr.left, scope),
                    self._translate_expr(expr.right, scope),
                ],
            )
        if op in ("+", "-", "*", "/"):
            # Integer-vs-real division (Req 13.5 / 13.6): the type
            # affinity of the operands determines the result type. The
            # DRC ``ArithmeticNode`` does not carry a type annotation,
            # so the distinction is preserved implicitly through the
            # operand types — cvc5's SMT translation infers the correct
            # division operator from the SMT sorts of the operands.
            return ArithmeticNode(
                operator=op,  # type: ignore[arg-type]
                left=self._translate_expr(expr.left, scope),
                right=self._translate_expr(expr.right, scope),
            )
        raise self._fail_unsupported(
            f"binary_op_{op}",
            f"unsupported binary operator {op!r}",
            expr.pos,
        )

    def _translate_unary(
        self, expr: sql_ast.UnaryOp, scope: Scope
    ) -> DRCCondition:
        op = expr.op
        if op == "NOT":
            inner = self._translate_expr(expr.operand, scope)
            return NotNode(operand=inner)
        if op in ("+", "-"):
            # Unary plus is a no-op; unary minus subtracts from zero so
            # the DRC printer can render it without a dedicated node.
            inner = self._translate_expr(expr.operand, scope)
            if op == "+":
                return inner
            return ArithmeticNode(
                operator="-",
                left=LiteralNode(value=0, data_type="number"),
                right=inner,
            )
        if op in ("IS_NULL", "IS_NOT_NULL"):
            # ``col IS NULL`` / ``col IS NOT NULL`` — model as an
            # uninterpreted Bool predicate so both sides of an
            # equivalence check share the same symbol. Semantically
            # opaque to cvc5, but structurally consistent.
            inner = self._translate_expr(expr.operand, scope)
            return FunctionCallNode(function=op, arguments=[inner])
        raise self._fail_unsupported(
            f"unary_op_{op}",
            f"unsupported unary operator {op!r}",
            expr.pos,
        )

    def _try_arithmetic_aggregate(
        self, expr: "sql_ast.Expression", scope: "Scope"
    ):
        """If ``expr`` is an arithmetic operation whose both sides are
        aggregates, return an ``ArithmeticAggregateVariable``. Otherwise
        return ``None``.

        Handles patterns like ``COUNT(T1.Id) / COUNT(DISTINCT T2.Name)``
        and ``CAST(COUNT(x) AS REAL) / COUNT(y)`` (CAST is already
        stripped by the parser).
        """
        from text_to_sql_planner.types.drc import ArithmeticAggregateVariable

        if not isinstance(expr, sql_ast.BinaryOp):
            return None
        if expr.op not in ("+", "-", "*", "/"):
            return None

        left_agg = self._try_extract_aggregate(expr.left, scope)
        right_agg = self._try_extract_aggregate(expr.right, scope)
        if left_agg is None or right_agg is None:
            return None

        return ArithmeticAggregateVariable(
            operator=expr.op,  # type: ignore[arg-type]
            left=left_agg,
            right=right_agg,
        )

    def _try_extract_aggregate(
        self, expr: "sql_ast.Expression", scope: "Scope"
    ):
        """If ``expr`` is an Aggregate node, return the corresponding
        ``AggregateVariable``. Otherwise return ``None``."""
        if isinstance(expr, sql_ast.Aggregate):
            column_name = "*"
            if isinstance(expr.column, sql_ast.ColumnRef):
                resolved = self._translate_column_ref(expr.column, scope)
                if isinstance(resolved, VariableRefNode):
                    column_name = resolved.name
                else:
                    column_name = expr.column.name
            return AggregateVariable(
                function=expr.function,  # type: ignore[arg-type]
                column=column_name,
            )
        # Could be a number literal (e.g., * 100 or * 1.0) — treat as
        # a "trivial aggregate" by wrapping it. Not supported yet.
        return None

    def _translate_function_call(
        self, expr: sql_ast.FunctionCall, scope: Scope
    ) -> DRCCondition:
        """Translate a generic function call.

        ``strftime`` is given a dedicated row in the design's translation
        table (Req 13.7): the function name is preserved exactly and
        argument order/arity are passed through unchanged. Every other
        function name is also passed through; the planner's downstream
        consumers decide whether the function is supported. The
        translator does not police the function-name allow-list because
        BIRD gold queries call into a long tail of SQLite built-ins.
        """

        args = [self._translate_expr(a, scope) for a in expr.args]
        return FunctionCallNode(function=expr.name, arguments=args)

    def _translate_in_list(
        self, expr: sql_ast.InList, scope: Scope
    ) -> DRCCondition:
        """``c IN (lit1, lit2, ...)`` → disjunction of equalities.

        Each value is translated as an expression so that ``c IN
        (other_col, ...)`` (rare but legal) still works.
        """

        left = self._translate_expr(expr.left, scope)
        clauses: list[DRCCondition] = []
        for v in expr.values:
            right = self._translate_expr(v, scope)
            clauses.append(ComparisonNode(operator="=", left=left, right=right))
        return _disjoin(clauses)

    def _translate_in_subquery(
        self, expr: sql_ast.InSubquery, scope: Scope
    ) -> DRCCondition:
        """``c IN (SELECT s FROM ...)`` → ``∃ inner_vars. body ∧ (c = s)``.

        The inner SELECT's first result variable is treated as the
        comparison RHS; multi-column ``IN`` is not supported by the
        SQL grammar so this is unambiguous in our subset.
        """

        inner_query, inner_scope = self.translate_select(
            expr.subquery, parent=scope
        )
        inner_drc = _strip_wrappers(inner_query)
        if not inner_drc.result_variables:
            # ``SELECT *`` with no schema; impossible to match.
            raise self._fail_unsupported(
                "in_subquery_empty_projection",
                "IN subquery has no projected column",
                expr.pos,
            )
        first = inner_drc.result_variables[0]
        rhs_name = first.name if isinstance(first, ColumnVariable) else first.column
        inner_bound = list(inner_scope.bound_vars.values())
        outer = self._translate_expr(expr.left, scope)
        eq = ComparisonNode(
            operator="=",
            left=outer,
            right=VariableRefNode(name=rhs_name),
        )
        body = _conjoin([inner_drc.condition, eq])
        return QuantifierNode(kind="exists", variables=inner_bound, body=body)

    def _translate_exists(
        self, expr: sql_ast.ExistsExpr, scope: Scope
    ) -> DRCCondition:
        """``EXISTS (SELECT ...)`` → ``∃ inner_vars. body``.

        ``NOT EXISTS`` is parsed as ``UnaryOp("NOT", ExistsExpr)`` and
        therefore wraps the result of this method in a ``NotNode`` at
        the caller level — see the design's translation table.
        """

        inner_query, inner_scope = self.translate_select(
            expr.subquery, parent=scope
        )
        inner_drc = _strip_wrappers(inner_query)
        inner_bound = list(inner_scope.bound_vars.values())
        return QuantifierNode(
            kind="exists", variables=inner_bound, body=inner_drc.condition
        )

    # ----- SELECT list / GROUP BY / ORDER BY -----------------------------

    def _translate_select_list(
        self, stmt: sql_ast.SelectStatement, scope: Scope
    ) -> list[ResultVariable]:
        """Project select-list items into ``result_variables``.

        Plain column references emit a :class:`ColumnVariable`.
        Aggregates emit an :class:`AggregateVariable`. ``*`` expands to
        every column of every table in scope, in FROM order. ``GROUP BY``
        keys appear alongside the aggregate result variables (the planner
        reads the group keys back from the non-aggregate entries).
        """

        out: list[ResultVariable] = []
        seen_names: set[str] = set()

        def add_column_by_alias(alias: str, col: str, pos: sql_ast.Position) -> None:
            entry = scope.bound_vars.get((alias.lower(), col.lower()))
            if entry is None:
                raise self._fail_unbound(
                    f"unbound column reference: {alias}.{col}", pos
                )
            if entry not in seen_names:
                seen_names.add(entry)
                out.append(ColumnVariable(name=entry))

        # Track GROUP BY keys so we can emit them as ``ColumnVariable``s
        # next to the aggregates even when they don't appear in the
        # select list. Per the design, aggregates and group keys live
        # side-by-side in ``result_variables``.
        group_alias_cols: list[tuple[str, str]] = []
        for g in stmt.group_by:
            ga = (g.qualifier or _single_alias_or_none(scope) or "").lower()
            group_alias_cols.append((ga, g.name.lower()))

        for item in stmt.select_list:
            expr = item.expr
            # ``SELECT *`` shorthand.
            if isinstance(expr, sql_ast.Literal) and expr.value == "*" and expr.data_type == "string":
                for alias_l, cols in scope.columns_by_alias.items():
                    for c in cols:
                        add_column_by_alias(alias_l, c, expr.pos)
                continue

            if isinstance(expr, sql_ast.ColumnRef):
                # Resolve the column to its bound variable name, applying
                # the same fallback rules as expression translation so
                # quoted-token-as-literal still works.
                resolved = self._translate_column_ref(expr, scope)
                if isinstance(resolved, VariableRefNode):
                    if resolved.name not in seen_names:
                        seen_names.add(resolved.name)
                        out.append(ColumnVariable(name=resolved.name))
                else:
                    # Fallback string literal — keep the original name as
                    # the projected column. This preserves the surface
                    # appearance of e.g. SELECT "Yes" FROM ... gold queries.
                    fallback_name = expr.name
                    if fallback_name not in seen_names:
                        seen_names.add(fallback_name)
                        out.append(ColumnVariable(name=fallback_name))
                continue

            if isinstance(expr, sql_ast.Aggregate):
                column_name = "*"
                if isinstance(expr.column, sql_ast.ColumnRef):
                    resolved = self._translate_column_ref(expr.column, scope)
                    if isinstance(resolved, VariableRefNode):
                        column_name = resolved.name
                    else:
                        column_name = expr.column.name
                elif isinstance(expr.column, sql_ast.CaseExpr):
                    # Conditional aggregation: COUNT(CASE WHEN ... THEN col END)
                    from text_to_sql_planner.types.drc import ConditionalAggregateVariable
                    case = expr.column
                    # Translate the WHEN condition.
                    cond = self._translate_expr(case.when_condition, scope)
                    # The THEN expression should be a column reference.
                    col_name = "expr"
                    if isinstance(case.then_expr, sql_ast.ColumnRef):
                        resolved = self._translate_column_ref(case.then_expr, scope)
                        if isinstance(resolved, VariableRefNode):
                            col_name = resolved.name
                        else:
                            col_name = case.then_expr.name
                    out.append(
                        ConditionalAggregateVariable(
                            function=expr.function,  # type: ignore[arg-type]
                            column=col_name,
                            condition=cond,
                        )
                    )
                    continue
                # ``Literal`` here is the ``COUNT(*)`` shorthand.
                out.append(
                    AggregateVariable(
                        function=expr.function,  # type: ignore[arg-type]
                        column=column_name,
                    )
                )
                continue

            # Computed expressions (concat, arithmetic, function calls)
            # in the SELECT list don't have a clean column-variable
            # name. Check if it's an arithmetic of two aggregates first
            # (the dev_556 pattern: COUNT(x) / COUNT(DISTINCT y)).
            arith_agg = self._try_arithmetic_aggregate(expr, scope)
            if arith_agg is not None:
                out.append(arith_agg)
                continue

            # Otherwise emit a synthetic ``ColumnVariable`` so the
            # projection at least has the right arity.
            synth = item.alias or f"expr_{len(out) + 1}"
            if synth not in seen_names:
                seen_names.add(synth)
                out.append(ColumnVariable(name=synth))

        # Append GROUP BY keys that the SELECT list omitted, so the
        # planner reads them back as group keys.
        for alias_l, col_l in group_alias_cols:
            entry = scope.bound_vars.get((alias_l, col_l))
            if entry is None:
                # Unbound group key: skip rather than crash; the WHERE
                # / select-list translation already errors for unbound
                # refs, so by the time we get here the key has been
                # translated successfully.
                continue
            if entry not in seen_names:
                seen_names.add(entry)
                out.append(ColumnVariable(name=entry))

        return out

    def _translate_order_key(
        self, key: sql_ast.OrderKey, scope: Scope
    ) -> SortCriterion:
        """Translate an ORDER BY key into a :class:`SortCriterion`.

        The DRC ``SortCriterion`` only has slots for ``column`` (a name)
        and an optional ``aggregate`` function. Bare column refs land
        directly; aggregates are split into ``column`` + ``aggregate``;
        any other expression collapses into a synthetic name (matching
        the SELECT-list behaviour above).
        """

        expr = key.expr
        if isinstance(expr, sql_ast.ColumnRef):
            resolved = self._translate_column_ref(expr, scope)
            name = (
                resolved.name
                if isinstance(resolved, VariableRefNode)
                else expr.name
            )
            return SortCriterion(column=name, direction=key.direction)
        if isinstance(expr, sql_ast.Aggregate):
            col_name = "*"
            if isinstance(expr.column, sql_ast.ColumnRef):
                resolved = self._translate_column_ref(expr.column, scope)
                col_name = (
                    resolved.name
                    if isinstance(resolved, VariableRefNode)
                    else expr.column.name
                )
            return SortCriterion(
                column=col_name,
                direction=key.direction,
                aggregate=expr.function,  # type: ignore[arg-type]
            )
        # Synthetic name for computed expressions.
        return SortCriterion(column="expr", direction=key.direction)


# --- Module-level helpers --------------------------------------------------


def _single_alias_or_none(scope: Scope) -> str | None:
    """Return the single table alias in ``scope`` if there is exactly one."""
    aliases = list(scope.tables.keys())
    if len(aliases) == 1:
        return aliases[0]
    return None


def _strip_wrappers(query: QueryExpression) -> DRCExpression:
    """Peel ``LIMIT``/``ORDER BY`` wrappers off a query expression."""
    while not isinstance(query, DRCExpression):
        query = query.inner
    return query


# --- Public entry point ----------------------------------------------------


def translate(
    stmt: sql_ast.SelectStatement, schema: str
) -> QueryExpression | ConverterError:
    """Translate a parsed SELECT into a DRC :class:`QueryExpression`.

    Returns the query expression on success, or a ``ConverterError`` on
    any unsupported-feature, unbound-reference, or other translation
    failure. Never raises for malformed input — the BIRD suite relies on
    the structured-error channel to keep running after a single
    Test_Case's gold SQL fails to translate.
    """

    schema_map = parse_schema(schema)
    translator = _Translator(schema_map)
    try:
        query, _scope = translator.translate_select(stmt, parent=None)
    except _TranslateFail as exc:
        return exc.error
    return query


# Public alias matching the name used in design.md's translation table and
# the task description for task 5.1 ("Implement the public translate
# function as ``translate_select(stmt, schema)``"). ``translate`` is kept
# as the primary entry point for callers that want a shorter name.
translate_select = translate


__all__ = ["translate", "translate_select", "Scope"]
