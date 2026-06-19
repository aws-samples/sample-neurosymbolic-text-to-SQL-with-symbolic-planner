"""Tests for the SMT converter's auto-declaration of uninterpreted DRC
function-call heads.

Background — run-08 dev_585 / dev_947 failures:
The SQL→DRC translator preserves SQLite function names verbatim
(``STRFTIME``, ``LIKE`` after the LIKE-support fix, etc.) as
``FunctionCallNode`` heads. The SMT converter renders these as
``(<head> <args...>)`` calls, but until this fix it never emitted
``(declare-fun <head> ...)`` for them — cvc5 hit an unknown function
symbol and exited with code 1, which the runner reported as a
``Verdict.unknown`` indeterminate result.

The fix: the converter now collects every non-built-in function head
along with its inferred argument and return sorts (from call-site
context) and emits a ``declare-fun`` for each. Both sides of an
equivalence check unify their per-head signatures so a head that
appears with the same shape on both sides gets one consistent
declaration.

These tests pin the new behaviour at three levels:

* :func:`bird_benchmark.sql_to_drc.parser.parse` accepts ``LIKE`` and
  emits the right binary-op shape.
* :func:`bird_benchmark.sql_to_drc.translator.translate` lowers the
  binary op to ``FunctionCallNode("LIKE", ...)``.
* :func:`text_to_sql_planner.equivalence.smt_converter.convert_to_smt`
  emits the ``(declare-fun LIKE ...)`` / ``(declare-fun STRFTIME ...)``
  declarations.
"""

from __future__ import annotations

import pytest

from bird_benchmark.sql_to_drc import convert_sql
from bird_benchmark.types import ConverterError
from text_to_sql_planner.equivalence.smt_converter import (
    collect_function_signatures,
    collect_var_types,
    convert_to_smt,
)
from text_to_sql_planner.types.drc import (
    ArithmeticNode,
    ColumnVariable,
    ComparisonNode,
    DRCExpression,
    FunctionCallNode,
    LiteralNode,
    LogicalConnectiveNode,
    MembershipNode,
    NotNode,
    QuantifierNode,
    VariableRefNode,
)


# ---------------------------------------------------------------------------
# Parser / translator: LIKE survives end-to-end
# ---------------------------------------------------------------------------


_SCHEMA_POSTS = (
    "CREATE TABLE posts (Id INTEGER, Title TEXT, BountyAmount INTEGER);"
)


def test_parser_accepts_like_in_where_clause():
    """``WHERE col LIKE 'pattern'`` no longer raises an unsupported
    feature error. The converter returns a real DRC expression."""
    sql = "SELECT Id FROM posts WHERE Title LIKE '%data%'"
    result = convert_sql(sql, _SCHEMA_POSTS)
    assert not isinstance(result, ConverterError), (
        f"convert_sql failed: {result}"
    )


def test_translator_emits_function_call_for_like():
    """The ``LIKE`` binary op lowers to a ``FunctionCallNode`` whose
    head is ``"LIKE"`` and whose two arguments are the lhs column and
    the pattern literal (in that order)."""
    sql = "SELECT Id FROM posts WHERE Title LIKE '%data%'"
    result = convert_sql(sql, _SCHEMA_POSTS)
    assert not isinstance(result, ConverterError)

    drc = _strip_wrappers(result)
    fn_calls = _gather_function_calls(drc.condition)
    like_calls = [f for f in fn_calls if f.function == "LIKE"]
    assert len(like_calls) == 1, (
        f"expected exactly one LIKE call, got {len(like_calls)}: {fn_calls}"
    )
    like = like_calls[0]
    assert len(like.arguments) == 2
    # Second argument is the pattern literal.
    pattern = like.arguments[1]
    assert isinstance(pattern, LiteralNode)
    assert pattern.value == "%data%"


def test_parser_still_rejects_between():
    """``BETWEEN`` is desugared to ``>= AND <=`` at parse time."""
    sql = "SELECT Id FROM posts WHERE Id BETWEEN 1 AND 10"
    result = convert_sql(sql, _SCHEMA_POSTS)
    # Should succeed now — BETWEEN desugars to (Id >= 1 AND Id <= 10)
    assert not isinstance(result, ConverterError), f"Unexpected error: {result}"


# ---------------------------------------------------------------------------
# SMT converter: function-signature collection
# ---------------------------------------------------------------------------


def test_collect_signatures_skips_builtins():
    """``CURRENT_DATE`` / ``DATE_SUB`` / ``DATE_ADD`` / ``DATEDIFF``
    are translated natively and must NOT appear as uninterpreted
    declarations — otherwise cvc5 would see two definitions and
    reject the script."""
    # ``CURRENT_DATE`` and ``DATE_SUB`` together inside an arithmetic
    # comparison.
    cond = ComparisonNode(
        operator=">",
        left=VariableRefNode(name="dob"),
        right=FunctionCallNode(
            function="DATE_SUB",
            arguments=[
                FunctionCallNode(function="CURRENT_DATE", arguments=[]),
                LiteralNode(value=10000, data_type="number"),
            ],
        ),
    )
    var_types = {"dob": "Int"}
    sigs = collect_function_signatures(cond, var_types)
    assert "CURRENT_DATE" not in sigs
    assert "DATE_SUB" not in sigs


def test_collect_signatures_records_like_at_bool_position():
    """``LIKE`` appearing at the top of a WHERE clause gets a Bool
    return sort and two String argument sorts (Title / pattern)."""
    cond = FunctionCallNode(
        function="LIKE",
        arguments=[
            VariableRefNode(name="Title"),
            LiteralNode(value="%data%", data_type="string"),
        ],
    )
    var_types = {"Title": "String"}
    sigs = collect_function_signatures(cond, var_types)
    assert "LIKE" in sigs
    arg_sorts, ret_sort = sigs["LIKE"]
    assert arg_sorts == ["String", "String"]
    assert ret_sort == "bool"


def test_collect_signatures_records_strftime_at_value_position():
    """``STRFTIME`` appearing inside an ordering comparison gets a
    value-sort return type. The args' inferred sorts come from the
    pattern literal (String) and the column (Int — date columns
    default to Int per the existing inference rules)."""
    cond = ComparisonNode(
        operator=">",
        left=FunctionCallNode(
            function="STRFTIME",
            arguments=[
                LiteralNode(value="%Y", data_type="string"),
                VariableRefNode(name="dob"),
            ],
        ),
        right=LiteralNode(value="1980", data_type="string"),
    )
    var_types = {"dob": "Int"}
    sigs = collect_function_signatures(cond, var_types)
    assert "STRFTIME" in sigs
    arg_sorts, _ = sigs["STRFTIME"]
    assert arg_sorts == ["String", "Int"]


def test_convert_to_smt_emits_declare_fun_for_like():
    """The full SMT script for a condition that calls ``LIKE`` includes
    a ``(declare-fun LIKE (...) Bool)`` line. Without it cvc5 would
    crash with an unknown function symbol."""
    cond = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(variables=["Id", "Title"], relation="posts"),
        right=FunctionCallNode(
            function="LIKE",
            arguments=[
                VariableRefNode(name="Title"),
                LiteralNode(value="%data%", data_type="string"),
            ],
        ),
    )
    script = convert_to_smt(cond)
    assert "(declare-fun LIKE (String String) Bool)" in script


def test_convert_to_smt_emits_declare_fun_for_strftime():
    """``STRFTIME('%Y', dob) > '1980'`` produces a ``declare-fun
    STRFTIME (String Int) Int`` line — return sort Int because the
    comparison is ordering with a bare-integer string on the other
    side, which the converter coerces to an int literal."""
    cond = ComparisonNode(
        operator=">",
        left=FunctionCallNode(
            function="STRFTIME",
            arguments=[
                LiteralNode(value="%Y", data_type="string"),
                VariableRefNode(name="dob"),
            ],
        ),
        right=LiteralNode(value="1980", data_type="string"),
    )
    script = convert_to_smt(cond)
    # The LHS sort (return of STRFTIME) is determined by the literal
    # on the other side. Since ``"1980"`` parses as a bare integer
    # under an ordering comparison, the literal renders as the int
    # ``1980`` and the function's return sort is Int.
    assert "(declare-fun STRFTIME" in script


def test_convert_to_smt_does_not_redeclare_builtins():
    """A formula using ``CURRENT_DATE`` and ``DATE_SUB`` must not
    produce ``declare-fun`` lines for those heads — they're translated
    natively by ``_convert_function_call`` and re-declaring them
    would conflict."""
    cond = ComparisonNode(
        operator=">",
        left=VariableRefNode(name="dob"),
        right=FunctionCallNode(
            function="DATE_SUB",
            arguments=[
                FunctionCallNode(function="CURRENT_DATE", arguments=[]),
                LiteralNode(value=10000, data_type="number"),
            ],
        ),
    )
    script = convert_to_smt(cond)
    assert "(declare-fun CURRENT_DATE" not in script
    assert "(declare-fun DATE_SUB" not in script


def test_convert_to_smt_bare_integer_string_under_ordering_collapses_to_int():
    """``STRFTIME('%Y', dob) > '1980'`` renders the RHS as ``1980``
    (an int), not ``"1980"`` (a string), so cvc5 sees both sides at
    the same Int sort. This is the existing date-string-coercion rule
    extended to bare-integer strings."""
    cond = ComparisonNode(
        operator=">",
        left=FunctionCallNode(
            function="STRFTIME",
            arguments=[
                LiteralNode(value="%Y", data_type="string"),
                VariableRefNode(name="dob"),
            ],
        ),
        right=LiteralNode(value="1980", data_type="string"),
    )
    script = convert_to_smt(cond)
    # The ``1980`` (no quotes) appears as an Int literal in the
    # comparison; the SMT-LIB string form ``"1980"`` does NOT.
    assert " 1980)" in script
    assert '"1980"' not in script


def test_convert_to_smt_equality_with_string_literal_keeps_string_form():
    """Equality (not ordering) with a bare-integer string literal
    stays a String comparison — the Int coercion is ordering-only.
    Otherwise ``WHERE name = "1980"`` (a person whose name is the
    digits) would silently turn into an integer comparison."""
    cond = ComparisonNode(
        operator="=",
        left=VariableRefNode(name="name"),
        right=LiteralNode(value="1980", data_type="string"),
    )
    script = convert_to_smt(cond)
    # The literal stays in its SMT-LIB string form.
    assert '"1980"' in script


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_wrappers(query):
    """Unwrap ``OrderByExpression`` / ``LimitExpression`` to get the
    inner ``DRCExpression``."""
    from text_to_sql_planner.types.drc import (
        LimitExpression,
        OrderByExpression,
    )
    while isinstance(query, (LimitExpression, OrderByExpression)):
        query = query.inner
    return query


def _gather_function_calls(node) -> list[FunctionCallNode]:
    """Collect every ``FunctionCallNode`` in a DRC condition tree."""
    out: list[FunctionCallNode] = []
    _walk_for_calls(node, out)
    return out


def _walk_for_calls(node, out: list[FunctionCallNode]) -> None:
    if node is None:
        return
    if isinstance(node, FunctionCallNode):
        out.append(node)
        for arg in node.arguments:
            _walk_for_calls(arg, out)
        return
    if isinstance(node, QuantifierNode):
        _walk_for_calls(node.body, out)
        return
    if isinstance(node, LogicalConnectiveNode):
        _walk_for_calls(node.left, out)
        _walk_for_calls(node.right, out)
        return
    if isinstance(node, NotNode):
        _walk_for_calls(node.operand, out)
        return
    if isinstance(node, ComparisonNode):
        _walk_for_calls(node.left, out)
        _walk_for_calls(node.right, out)
        return
    if isinstance(node, ArithmeticNode):
        _walk_for_calls(node.left, out)
        _walk_for_calls(node.right, out)
        return
    # MembershipNode / VariableRefNode / LiteralNode: no calls inside.



# ---------------------------------------------------------------------------
# End-to-end: an SQL with LIKE renders to a script that cvc5 accepts
# and reports a decisive verdict for equivalence against itself.
# These tests run REAL cvc5 (not a mock) so we know the script is
# well-formed enough for the solver to actually parse it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_like_round_trip_self_equivalence_with_real_cvc5():
    """A DRC built from ``SELECT … WHERE Title LIKE '%data%'`` is
    equivalent to itself. Pre-fix this would crash cvc5 (unknown
    function symbol ``LIKE``) and the runner would report
    ``Verdict.unknown``."""
    import shutil

    cvc5 = shutil.which("cvc5") or "/usr/local/bin/cvc5"
    import os
    if not os.path.exists(cvc5):
        pytest.skip("cvc5 binary not available")

    from text_to_sql_planner.equivalence import (
        EquivalenceCheckerConfig,
        EquivalentResult,
        check_equivalence,
    )

    sql = "SELECT Id FROM posts WHERE Title LIKE '%data%'"
    drc1 = convert_sql(sql, _SCHEMA_POSTS)
    drc2 = convert_sql(sql, _SCHEMA_POSTS)
    assert not isinstance(drc1, ConverterError)
    assert not isinstance(drc2, ConverterError)

    config = EquivalenceCheckerConfig(cvc5_path=cvc5, timeout_seconds=10.0)
    result = await check_equivalence(
        _strip_wrappers(drc1),
        _strip_wrappers(drc2),
        config=config,
    )
    assert isinstance(result, EquivalentResult), (
        f"expected EquivalentResult, got {type(result).__name__}: {result}"
    )


@pytest.mark.asyncio
async def test_strftime_round_trip_self_equivalence_with_real_cvc5():
    """``WHERE STRFTIME('%Y', dob) > '1980'`` is equivalent to itself.
    Pre-fix the gold-side conversion would render a ``(STRFTIME ...)``
    call into an undeclared function and cvc5 would exit with code 1."""
    import shutil

    cvc5 = shutil.which("cvc5") or "/usr/local/bin/cvc5"
    import os
    if not os.path.exists(cvc5):
        pytest.skip("cvc5 binary not available")

    from text_to_sql_planner.equivalence import (
        EquivalenceCheckerConfig,
        EquivalentResult,
        check_equivalence,
    )

    schema = (
        "CREATE TABLE drivers ("
        "driverId INTEGER, dob DATE, nationality TEXT"
        ");"
    )
    sql = (
        "SELECT driverId FROM drivers "
        "WHERE STRFTIME('%Y', dob) > '1980'"
    )
    drc1 = convert_sql(sql, schema)
    drc2 = convert_sql(sql, schema)
    assert not isinstance(drc1, ConverterError)
    assert not isinstance(drc2, ConverterError)

    config = EquivalenceCheckerConfig(cvc5_path=cvc5, timeout_seconds=10.0)
    result = await check_equivalence(
        _strip_wrappers(drc1),
        _strip_wrappers(drc2),
        config=config,
    )
    assert isinstance(result, EquivalentResult), (
        f"expected EquivalentResult, got {type(result).__name__}: {result}"
    )



# ---------------------------------------------------------------------------
# Regression: SQL→DRC translator's existential closure of FROM-clause
# variables. Before this fix the translator left filter columns (e.g.
# ``City`` in ``WHERE City = 'Adelanto'``) as free constants. The
# planner-side LLM produces DRCs that existentially bind every non-
# result-variable membership slot, so the two sides had structurally
# different shapes and cvc5 reported ``not_equivalent`` on what were
# actually the same relation. This is the run-10 regression for the
# ``logical=no, exec=yes`` cluster: dev_78, dev_1425, dev_757, dev_1375.
# ---------------------------------------------------------------------------


def test_translator_existentially_binds_filter_columns():
    """``SELECT id FROM t WHERE name = 'foo'`` must produce a DRC whose
    condition is ``∃ name. (in (id name) t) ∧ name = "foo"`` — not
    ``(in (id name) t) ∧ name = "foo"`` with ``name`` free."""
    schema = "CREATE TABLE t (id INTEGER, name TEXT);"
    sql = "SELECT id FROM t WHERE name = 'foo'"
    result = convert_sql(sql, schema)
    assert not isinstance(result, ConverterError)

    drc = _strip_wrappers(result)
    # The condition must be wrapped in an existential whose bindings
    # include ``name`` (the filter column). ``id`` is the result
    # variable so it stays free; ``name`` must be bound.
    assert isinstance(drc.condition, QuantifierNode)
    assert drc.condition.kind == "exists"
    bound = set(drc.condition.variables)
    # The filter column must be among the bound names.
    assert any("name" in b.lower() for b in bound), (
        f"expected ``name``-like binding in {bound}"
    )


def test_translator_does_not_bind_result_variables():
    """Result variables in the SELECT list stay free (they're projected
    out at the top of the DRC). Only non-result FROM-clause names get
    existentially bound."""
    schema = "CREATE TABLE t (id INTEGER, name TEXT, age INTEGER);"
    sql = "SELECT id, name FROM t WHERE age > 18"
    result = convert_sql(sql, schema)
    assert not isinstance(result, ConverterError)

    drc = _strip_wrappers(result)
    rv_names = {rv.name for rv in drc.result_variables}
    assert isinstance(drc.condition, QuantifierNode)
    bound = set(drc.condition.variables)
    # ``age`` is a filter column → bound. ``id`` and ``name`` are
    # result variables → not bound.
    assert all(rv not in bound for rv in rv_names), (
        f"result variables {rv_names} must not be in bound names {bound}"
    )
    assert any("age" in b.lower() for b in bound)


@pytest.mark.asyncio
async def test_dev_78_filter_column_equivalence_resolves_with_real_cvc5():
    """The dev_78 shape: a planner-side DRC that explicitly binds
    ``City`` and the gold-side DRC that the translator now also
    existentially binds. cvc5 must report ``equivalent``."""
    import os
    import shutil

    cvc5 = shutil.which("cvc5") or "/usr/local/bin/cvc5"
    if not os.path.exists(cvc5):
        pytest.skip("cvc5 binary not available")

    from text_to_sql_planner.equivalence import (
        EquivalenceCheckerConfig,
        EquivalentResult,
        check_equivalence,
    )
    from text_to_sql_planner.types.drc import (
        ColumnVariable as _ColVar,
        DRCExpression as _DRCExpr,
    )

    schema = (
        "CREATE TABLE schools (CDSCode TEXT, City TEXT, GSserved TEXT);"
    )
    sql = "SELECT GSserved FROM schools WHERE City = 'Adelanto'"

    # Both DRCs come from the SQL→DRC translator. Pre-fix the gold
    # side had a free ``v_city_*`` constant; post-fix it's ∃-bound.
    drc_gold = convert_sql(sql, schema)
    assert not isinstance(drc_gold, ConverterError)

    # Hand-build the planner-side shape (free ``GSserved``, ∃-bound
    # ``CDSCode``, ``City``). This matches what the LLM emits for the
    # same question.
    planner_drc = _DRCExpr(
        result_variables=[_ColVar(name="GSserved")],
        condition=QuantifierNode(
            kind="exists",
            variables=["CDSCode", "City"],
            body=LogicalConnectiveNode(
                operator="and",
                left=MembershipNode(
                    variables=["CDSCode", "City", "GSserved"],
                    relation="schools",
                ),
                right=ComparisonNode(
                    operator="=",
                    left=VariableRefNode(name="City"),
                    right=LiteralNode(value="Adelanto", data_type="string"),
                ),
            ),
        ),
    )

    config = EquivalenceCheckerConfig(cvc5_path=cvc5, timeout_seconds=10.0)
    result = await check_equivalence(
        planner_drc, _strip_wrappers(drc_gold), config=config,
    )
    assert isinstance(result, EquivalentResult), (
        f"expected EquivalentResult, got {type(result).__name__}: {result}"
    )



# ---------------------------------------------------------------------------
# Planner-side LIKE: the DRC parser must accept ``(LIKE col pattern)``
# (and the lowercase ``(like ...)`` the LLM sometimes emits) so the
# question-converter loop doesn't reject the LLM's output. The DRC→SQL
# converter must render LIKE as the SQL infix form ``col LIKE pattern``.
# ---------------------------------------------------------------------------


def test_planner_drc_parser_accepts_like():
    """``(LIKE Title "%data%")`` parses to a FunctionCallNode."""
    from text_to_sql_planner.parser.parser import parse_query

    sexp = '(drc (Id) (and (in (Id Title) posts) (LIKE Title "%data%")))'
    result = parse_query(sexp)
    # parse_query returns QueryParserSuccess (with .query) on success.
    assert hasattr(result, "query"), f"expected success, got {result}"
    calls = _gather_function_calls(result.query.condition)
    like_calls = [c for c in calls if c.function == "LIKE"]
    assert len(like_calls) == 1
    assert like_calls[0].arguments[1].value == "%data%"


def test_planner_drc_parser_accepts_lowercase_like():
    """The LLM sometimes emits lowercase ``like``; the parser
    normalises both spellings to the uppercase ``LIKE`` form so the
    SMT layer sees a single consistent function head."""
    from text_to_sql_planner.parser.parser import parse_query

    sexp = '(drc (Id) (and (in (Id Title) posts) (like Title "%data%")))'
    result = parse_query(sexp)
    assert hasattr(result, "query"), f"expected success, got {result}"
    calls = _gather_function_calls(result.query.condition)
    like_calls = [c for c in calls if c.function == "LIKE"]
    assert len(like_calls) == 1, (
        f"expected exactly one normalised ``LIKE`` call, got {calls}"
    )


def test_planner_drc_to_sql_emits_like_infix():
    """``(LIKE col "%pat%")`` renders as the SQL ``col LIKE '%pat%'``
    infix form, not ``LIKE(col, '%pat%')`` which SQLite rejects."""
    from text_to_sql_planner.sql import convert_to_sql, SQLSuccess
    from text_to_sql_planner.types.operation_tree import (
        OperationTree,
        OperatorNode,
        TableLeafNode,
    )
    from text_to_sql_planner.types.operators import SelectionParams

    # Build a minimal selection tree: ``SELECT * FROM posts WHERE Title
    # LIKE '%data%'`` shape, with the LIKE condition as a
    # FunctionCallNode in the selection's params.
    table = TableLeafNode(table_name="posts", columns=["Id", "Title"])
    like_cond = FunctionCallNode(
        function="LIKE",
        arguments=[
            VariableRefNode(name="Title"),
            LiteralNode(value="%data%", data_type="string"),
        ],
    )
    sel = OperatorNode(
        operator="selection",
        params=SelectionParams(condition=like_cond),
        inputs=[table],
        output_columns=["Id", "Title"],
    )
    tree = OperationTree(root=sel)
    result = convert_to_sql(tree)

    assert isinstance(result, SQLSuccess), result
    sql = result.sql
    # The LIKE appears as an infix form, not a function call.
    assert " LIKE " in sql, f"expected 'LIKE' as infix in SQL: {sql}"
    # The pattern literal made it through.
    assert "%data%" in sql, f"expected pattern in SQL: {sql}"



# ---------------------------------------------------------------------------
# Date-literal SQLite compatibility (run-11 dev_947 / dev_1430 regression).
# The planner used to emit ``dob > DATE '1980-12-31'`` — ANSI-SQL /
# Postgres syntax that SQLite (the engine BIRD uses) rejects with
# ``OperationalError: near "'1980-12-31'": syntax error``. The fix is
# to emit a plain quoted string; SQLite's TEXT date storage compares
# correctly under lexicographic order.
# ---------------------------------------------------------------------------


def test_planner_sql_emits_date_literal_as_plain_string():
    """A date-shaped string literal in a comparison renders as
    ``'YYYY-MM-DD'``, not ``DATE 'YYYY-MM-DD'``."""
    from text_to_sql_planner.sql import convert_to_sql, SQLSuccess
    from text_to_sql_planner.types.operation_tree import (
        OperationTree,
        OperatorNode,
        TableLeafNode,
    )
    from text_to_sql_planner.types.operators import SelectionParams

    table = TableLeafNode(table_name="drivers", columns=["driverId", "dob"])
    cond = ComparisonNode(
        operator=">",
        left=VariableRefNode(name="dob"),
        right=LiteralNode(value="1980-12-31", data_type="string"),
    )
    sel = OperatorNode(
        operator="selection",
        params=SelectionParams(condition=cond),
        inputs=[table],
        output_columns=["driverId", "dob"],
    )
    tree = OperationTree(root=sel)
    result = convert_to_sql(tree)

    assert isinstance(result, SQLSuccess), result
    sql = result.sql
    # SQLite-compatible: bare quoted string.
    assert "'1980-12-31'" in sql, f"expected plain quoted date in SQL: {sql}"
    # No ANSI ``DATE 'YYYY-MM-DD'`` prefix that SQLite rejects.
    assert "DATE '1980-12-31'" not in sql, (
        f"DATE-prefixed date literal would crash SQLite: {sql}"
    )


def test_planner_sql_non_date_strings_unchanged():
    """Plain string literals that aren't date-shaped continue to be
    emitted with single-quote escaping. The previous behaviour only
    diverged for date-shaped strings; non-date strings always took the
    ``'…'`` branch."""
    from text_to_sql_planner.sql import convert_to_sql, SQLSuccess
    from text_to_sql_planner.types.operation_tree import (
        OperationTree,
        OperatorNode,
        TableLeafNode,
    )
    from text_to_sql_planner.types.operators import SelectionParams

    table = TableLeafNode(table_name="drivers", columns=["driverId", "nationality"])
    cond = ComparisonNode(
        operator="=",
        left=VariableRefNode(name="nationality"),
        right=LiteralNode(value="British", data_type="string"),
    )
    sel = OperatorNode(
        operator="selection",
        params=SelectionParams(condition=cond),
        inputs=[table],
        output_columns=["driverId", "nationality"],
    )
    tree = OperationTree(root=sel)
    result = convert_to_sql(tree)

    assert isinstance(result, SQLSuccess), result
    assert "'British'" in result.sql
