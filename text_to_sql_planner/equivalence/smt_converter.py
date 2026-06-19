"""Convert DRC conditions to SMT-LIB syntax for cvc5."""

from __future__ import annotations

from text_to_sql_planner.types.drc import (
    ArithmeticNode,
    ComparisonNode,
    DRCCondition,
    FunctionCallNode,
    IsNotNullNode,
    LiteralNode,
    LogicalConnectiveNode,
    MembershipNode,
    NotNode,
    QuantifierNode,
    VariableRefNode,
)


def convert_to_smt(condition: DRCCondition) -> str:
    """Convert a DRC condition into a complete SMT-LIB script.

    Generates declarations for all relations as uninterpreted functions,
    asserts the formula, and includes (check-sat).
    Infers variable types from usage (String if compared to string literal, Int otherwise).
    """
    relations: dict[str, int] = {}
    variables: set[str] = set()
    var_types: dict[str, str] = {}  # variable -> "Int" or "String"
    _collect_symbols(condition, relations, variables)
    _infer_types(condition, var_types)

    lines: list[str] = []
    lines.append("(set-logic ALL)")

    # Declare all variables with inferred types
    for var in sorted(variables):
        sort = var_types.get(var, "Int")
        lines.append(f"(declare-const {var} {sort})")

    # Declare relations as uninterpreted functions returning Bool
    # Use mixed sorts based on the variables used in membership
    rel_sorts: dict[str, list[str]] = {}
    _collect_relation_sorts(condition, var_types, rel_sorts)
    for rel_name, arity in sorted(relations.items()):
        if rel_name in rel_sorts:
            sorts = " ".join(rel_sorts[rel_name])
        else:
            sorts = " ".join(["Int"] * arity)
        lines.append(f"(declare-fun {rel_name} ({sorts}) Bool)")

    # Declare any uninterpreted DRC function (e.g. ``LIKE``,
    # ``STRFTIME``) used by the formula. Without these
    # declarations cvc5 hits an unknown function symbol and exits
    # before attempting the proof.
    fn_sigs = collect_function_signatures(condition, var_types)
    for fn_name in sorted(fn_sigs):
        arg_sorts, ret_sort = fn_sigs[fn_name]
        ret_sort_smt = _smt_sort(ret_sort)
        sorts_str = " ".join(arg_sorts)
        lines.append(f"(declare-fun {fn_name} ({sorts_str}) {ret_sort_smt})")

    # Assert the formula
    formula = _convert_node(condition, var_types)
    lines.append(f"(assert {formula})")
    lines.append("(check-sat)")

    return "\n".join(lines)


def convert_condition_to_formula(condition: DRCCondition) -> str:
    """Convert a DRC condition to an SMT-LIB formula string (no script wrapper)."""
    var_types: dict[str, str] = {}
    _infer_types(condition, var_types)
    return _convert_node(condition, var_types)


def collect_symbols(condition: DRCCondition) -> tuple[dict[str, int], set[str]]:
    """Collect relation names (with arities) and variable names from a condition.

    Returns (relations_dict, variables_set).
    """
    relations: dict[str, int] = {}
    variables: set[str] = set()
    _collect_symbols(condition, relations, variables)
    return relations, variables


def collect_var_types(condition: DRCCondition) -> dict[str, str]:
    """Collect inferred variable types from a condition."""
    var_types: dict[str, str] = {}
    _infer_types(condition, var_types)
    return var_types


def _infer_types(node: DRCCondition, var_types: dict[str, str]) -> None:
    """Infer variable types from context (comparisons with literals).

    Rules:
    - Variable compared with = or != to a string literal → String (always wins)
    - Variable compared with <, >, <=, >= to a string literal → Int
      (dates/ordered strings are modeled as Int since SMT-LIB String doesn't support ordering)
    - Variable compared to a number → Int (only if not already String)
    - Variable used in arithmetic → Int (only if not already String)
    - String type always wins over Int (once a variable is seen as String, it stays String)
    """
    if node is None:
        return

    if isinstance(node, ComparisonNode):
        # Ordering comparisons force Int even for string literals (dates, etc.)
        is_ordering = node.operator in ("<", ">", "<=", ">=")

        if isinstance(node.left, VariableRefNode) and isinstance(node.right, LiteralNode):
            name = node.left.name
            if node.right.data_type == "string":
                if is_ordering:
                    # Dates and ordered strings → model as Int, but only if not already String
                    if var_types.get(name) != "String":
                        var_types[name] = "Int"
                else:
                    # String equality → always String (wins over Int)
                    var_types[name] = "String"
            elif node.right.data_type == "number":
                if var_types.get(name) != "String":
                    var_types.setdefault(name, "Int")
        elif isinstance(node.right, VariableRefNode) and isinstance(node.left, LiteralNode):
            name = node.right.name
            if node.left.data_type == "string":
                if is_ordering:
                    if var_types.get(name) != "String":
                        var_types[name] = "Int"
                else:
                    var_types[name] = "String"
            elif node.left.data_type == "number":
                if var_types.get(name) != "String":
                    var_types.setdefault(name, "Int")
        _infer_types(node.left, var_types)
        _infer_types(node.right, var_types)

    elif isinstance(node, LogicalConnectiveNode):
        _infer_types(node.left, var_types)
        _infer_types(node.right, var_types)

    elif isinstance(node, NotNode):
        _infer_types(node.operand, var_types)

    elif isinstance(node, QuantifierNode):
        _infer_types(node.body, var_types)

    elif isinstance(node, ArithmeticNode):
        # Variables in arithmetic are Int
        if isinstance(node.left, VariableRefNode):
            var_types.setdefault(node.left.name, "Int")
        if isinstance(node.right, VariableRefNode):
            var_types.setdefault(node.right.name, "Int")
        _infer_types(node.left, var_types)
        _infer_types(node.right, var_types)

    elif isinstance(node, FunctionCallNode):
        # ``LIKE(haystack, pattern)`` — both operands are SQL strings.
        # SQLite only matches LIKE on TEXT (or text-affinity) columns,
        # so any bare-variable operand is a String. Without this rule
        # ``Title`` in ``WHERE Title LIKE '%data%'`` would default to
        # Int and the LIKE declaration would mix sorts.
        if node.function == "LIKE":
            for arg in node.arguments:
                if isinstance(arg, VariableRefNode):
                    var_types[arg.name] = "String"
        for arg in node.arguments:
            _infer_types(arg, var_types)

    elif isinstance(node, IsNotNullNode):
        # ``IsNotNullNode`` does not pin a sort by itself. The column's
        # sort comes from elsewhere in the formula (a comparison with a
        # literal, a membership-slot site, etc.); when no other constraint
        # pins it, the Int default applies at conversion time.
        pass


def _collect_relation_sorts(
    node: DRCCondition,
    var_types: dict[str, str],
    rel_sorts: dict[str, list[str]],
) -> None:
    """Collect the sort signature for each relation based on variable types at membership sites.

    Per-position unification semantics
    -----------------------------------

    The same predicate can appear with different arities at different
    membership sites — e.g. a projection-style ``(in (a b) T)`` followed
    by a full-row ``(in (v1 v2 v3 v4) T)``. Both are valid uses of
    ``T`` (one consumer ignored some columns), but they let us see the
    sort of different positions.

    We unify across sites with three rules:

    1. The signature length is the *maximum* arity seen — that way
       full-row sites get a complete signature and short sites' info
       still applies to the prefix positions.
    2. Per position, ``String`` wins over ``Int`` when both are
       observed: a column whose value gets compared to a string
       literal anywhere in the query has to be declared ``String`` or
       cvc5 rejects the predicate call with a sort error.
    3. Positions only observed at one site (because they're past the
       end of a shorter site's tuple) take that site's sort.

    The previous "longest list wins" rule lost rule 2: a 9-arg
    membership of all-``Int`` would silently overwrite a 4-arg
    membership whose first two slots had been resolved to ``String``
    by literal-comparison inference. That's the bug behind dev_1519's
    ``cvc5(...) exited with code 1`` failure.
    """
    if node is None:
        return

    if isinstance(node, MembershipNode):
        site_sorts = [var_types.get(v, "Int") for v in node.variables]
        existing = rel_sorts.get(node.relation)
        if existing is None:
            rel_sorts[node.relation] = site_sorts
        else:
            # Per-position unification with String-wins semantics.
            unified_len = max(len(existing), len(site_sorts))
            unified: list[str] = []
            for i in range(unified_len):
                left = existing[i] if i < len(existing) else None
                right = site_sorts[i] if i < len(site_sorts) else None
                if left is None:
                    unified.append(right or "Int")
                elif right is None:
                    unified.append(left)
                elif left == "String" or right == "String":
                    unified.append("String")
                else:
                    unified.append(left)
            rel_sorts[node.relation] = unified

    elif isinstance(node, LogicalConnectiveNode):
        _collect_relation_sorts(node.left, var_types, rel_sorts)
        _collect_relation_sorts(node.right, var_types, rel_sorts)

    elif isinstance(node, NotNode):
        _collect_relation_sorts(node.operand, var_types, rel_sorts)

    elif isinstance(node, QuantifierNode):
        _collect_relation_sorts(node.body, var_types, rel_sorts)

    elif isinstance(node, ComparisonNode):
        _collect_relation_sorts(node.left, var_types, rel_sorts)
        _collect_relation_sorts(node.right, var_types, rel_sorts)

    elif isinstance(node, IsNotNullNode):
        # ``IsNotNullNode`` does not pin a relation slot's sort by
        # itself — the column's sort comes from elsewhere.
        pass


def _collect_symbols(
    node: DRCCondition,
    relations: dict[str, int],
    variables: set[str],
) -> None:
    """Recursively collect relation names (with arities) and variable names."""
    if isinstance(node, QuantifierNode):
        for v in node.variables:
            variables.add(v)
        if node.body is not None:
            _collect_symbols(node.body, relations, variables)

    elif isinstance(node, LogicalConnectiveNode):
        if node.left is not None:
            _collect_symbols(node.left, relations, variables)
        if node.right is not None:
            _collect_symbols(node.right, relations, variables)

    elif isinstance(node, NotNode):
        if node.operand is not None:
            _collect_symbols(node.operand, relations, variables)

    elif isinstance(node, ComparisonNode):
        if node.left is not None:
            _collect_symbols(node.left, relations, variables)
        if node.right is not None:
            _collect_symbols(node.right, relations, variables)

    elif isinstance(node, MembershipNode):
        for v in node.variables:
            variables.add(v)
        if node.relation:
            relations[node.relation] = max(
                relations.get(node.relation, 0), len(node.variables)
            )

    elif isinstance(node, IsNotNullNode):
        # ``IsNotNullNode.column`` is a column-binding name, treated the
        # same way as a ``MembershipNode`` slot variable: record it so it
        # gets a ``(declare-const)`` when free.
        if node.column:
            variables.add(node.column)

    elif isinstance(node, ArithmeticNode):
        if node.left is not None:
            _collect_symbols(node.left, relations, variables)
        if node.right is not None:
            _collect_symbols(node.right, relations, variables)

    elif isinstance(node, VariableRefNode):
        variables.add(node.name)

    elif isinstance(node, LiteralNode):
        pass  # No symbols to collect

    elif isinstance(node, FunctionCallNode):
        for arg in node.arguments:
            if arg is not None:
                _collect_symbols(arg, relations, variables)


def _convert_node(node: DRCCondition, var_types: dict[str, str] | None = None, scope: dict[str, str] | None = None) -> str:
    """Recursively convert a DRC condition node to SMT-LIB syntax.
    
    `scope` maps original variable names to their (possibly renamed) SMT-LIB names,
    handling shadowing in nested quantifiers.
    """
    if var_types is None:
        var_types = {}
    if scope is None:
        scope = {}
    if isinstance(node, QuantifierNode):
        return _convert_quantifier(node, var_types, scope)
    elif isinstance(node, LogicalConnectiveNode):
        return _convert_logical(node, var_types, scope)
    elif isinstance(node, NotNode):
        return _convert_not(node, var_types, scope)
    elif isinstance(node, ComparisonNode):
        return _convert_comparison(node, var_types, scope)
    elif isinstance(node, MembershipNode):
        return _convert_membership(node, scope)
    elif isinstance(node, ArithmeticNode):
        return _convert_arithmetic(node, var_types, scope)
    elif isinstance(node, LiteralNode):
        return _convert_literal(node)
    elif isinstance(node, FunctionCallNode):
        return _convert_function_call(node, var_types, scope)
    elif isinstance(node, VariableRefNode):
        # Use the renamed name if in scope
        return scope.get(node.name, node.name)
    elif isinstance(node, IsNotNullNode):
        # Mirror the SMT form ``rewrite_null_checks`` produces today
        # (a comparison against the empty-string / zero sentinel) so
        # gold-side ``IS_NOT_NULL(col)`` and planner-side
        # ``IsNotNullNode(col)`` collapse to the same SMT.
        col_name = scope.get(node.column, node.column)
        sort = var_types.get(node.column, "Int")
        if sort == "String":
            return f'(not (= {col_name} ""))'
        return f"(not (= {col_name} 0))"
    else:
        raise ValueError(f"Unknown DRC node type: {type(node)}")


_rename_counter: dict[str, int] = {}


def _fresh_name(base: str, all_names: set[str]) -> str:
    """Generate a fresh variable name that doesn't conflict with existing names."""
    if base not in all_names:
        return base
    counter = 2
    while f"{base}_{counter}" in all_names:
        counter += 1
    return f"{base}_{counter}"


def _convert_quantifier(node: QuantifierNode, var_types: dict[str, str], scope: dict[str, str]) -> str:
    quantifier = node.kind  # "forall" or "exists"

    # A quantifier-bound variable shadows the outer scope if its name
    # collides with either an outer key (an original name that's been
    # substituted) or an outer value (a name we substituted to). Both
    # cases must trigger alpha-rename, otherwise the body would either
    # accidentally drop an outer substitution (when ``v`` equals an outer
    # key) or capture an outer reference (when ``v`` equals an outer
    # value).
    outer_keys = set(scope.keys())
    outer_values = set(scope.values())
    forbidden = outer_keys | outer_values

    new_scope = dict(scope)  # copy outer scope
    renamed_vars: list[str] = []

    for v in node.variables:
        if v in forbidden:
            fresh = _fresh_name(v, forbidden)
            new_scope[v] = fresh
            forbidden.add(fresh)
            renamed_vars.append(fresh)
        else:
            new_scope[v] = v
            forbidden.add(v)
            renamed_vars.append(v)

    bindings = " ".join(f"({rv} {var_types.get(orig, 'Int')})"
                        for rv, orig in zip(renamed_vars, node.variables))
    body = _convert_node(node.body, var_types, new_scope)
    return f"({quantifier} ({bindings}) {body})"


def _convert_logical(node: LogicalConnectiveNode, var_types: dict[str, str], scope: dict[str, str]) -> str:
    left = _convert_node(node.left, var_types, scope)
    right = _convert_node(node.right, var_types, scope)
    op_map = {"and": "and", "or": "or", "implies": "=>"}
    op = op_map[node.operator]
    return f"({op} {left} {right})"


def _convert_not(node: NotNode, var_types: dict[str, str], scope: dict[str, str]) -> str:
    operand = _convert_node(node.operand, var_types, scope)
    return f"(not {operand})"


def _convert_comparison(node: ComparisonNode, var_types: dict[str, str], scope: dict[str, str]) -> str:
    is_ordering = node.operator in ("<", ">", "<=", ">=")

    left = _convert_node_for_comparison(node.left, var_types, is_ordering, scope)
    right = _convert_node_for_comparison(node.right, var_types, is_ordering, scope)

    op_map = {
        "=": "=",
        "!=": "distinct",
        "<": "<",
        ">": ">",
        "<=": "<=",
        ">=": ">=",
    }
    op = op_map[node.operator]
    return f"({op} {left} {right})"


def _convert_node_for_comparison(node: DRCCondition, var_types: dict[str, str], is_ordering: bool, scope: dict[str, str] | None = None) -> str:
    """Convert a node for use in a comparison, handling date literals."""
    if is_ordering and isinstance(node, LiteralNode) and node.data_type == "string":
        int_val = _date_string_to_int(str(node.value))
        if int_val is not None:
            return str(int_val)
        # Bare integer-shaped strings (e.g. ``"1980"``, ``"42"``) compared
        # under an ordering operator collapse to their integer value too.
        # SQLite's STRFTIME-vs-year-literal pattern (``STRFTIME('%Y', dob)
        # > '1980'``) lands here: with the LHS uninterpreted function
        # declared as ``Int``-returning, the RHS literal must also be
        # ``Int`` or cvc5 rejects the comparison with a sort mismatch.
        # We only do this for ordering operators because equality on
        # the same shape ``= "1980"`` is genuinely a string comparison.
        bare_int = _bare_int_string_to_int(str(node.value))
        if bare_int is not None:
            return str(bare_int)
    return _convert_node(node, var_types, scope)


def _bare_int_string_to_int(s: str) -> int | None:
    """Parse ``s`` as an integer, returning ``None`` if it isn't one.

    Accepts an optional leading ``-`` and rejects empty / whitespace-only
    inputs. Used by :func:`_convert_node_for_comparison` to coerce
    bare-integer string literals (``"1980"``) under ordering operators
    so they typecheck against an ``Int``-returning uninterpreted
    function on the other side.
    """
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _date_string_to_int(s: str) -> int | None:
    """Convert a date string like '2000-01-01' to days since epoch (1970-01-01).

    Returns None if the string doesn't look like a date.
    """
    import re
    from datetime import date

    # Match YYYY-MM-DD or YYYY/MM/DD
    m = re.match(r"^(\d{4})[-/](\d{2})[-/](\d{2})$", s)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            epoch = date(1970, 1, 1)
            return (d - epoch).days
        except ValueError:
            return None
    return None


def _convert_membership(node: MembershipNode, scope: dict[str, str] | None = None) -> str:
    if scope is None:
        scope = {}
    if not node.variables:
        # Zero-arity predicate (after slot pruning): render as a bare
        # propositional symbol rather than ``(R )``, which is invalid
        # SMT-LIB syntax. The corresponding ``declare-fun`` is also
        # written without arguments by the caller.
        return node.relation
    args = " ".join(scope.get(v, v) for v in node.variables)
    return f"({node.relation} {args})"


def _convert_arithmetic(node: ArithmeticNode, var_types: dict[str, str], scope: dict[str, str] | None = None) -> str:
    left = _convert_node(node.left, var_types, scope)
    right = _convert_node(node.right, var_types, scope)
    return f"({node.operator} {left} {right})"


def _convert_function_call(node: FunctionCallNode, var_types: dict[str, str], scope: dict[str, str] | None = None) -> str:
    """Convert a function call to SMT-LIB.

    CURRENT_DATE → today's epoch days (integer)
    DATE_SUB(x, n) → (- x n)
    DATE_ADD(x, n) → (+ x n)
    DATEDIFF(x, y) → (- x y)
    YEAR(x), MONTH(x), DAY(x) → treated as uninterpreted functions
    """
    from datetime import date

    if node.function == "CURRENT_DATE":
        today = date.today()
        epoch = date(1970, 1, 1)
        days = (today - epoch).days
        return str(days)
    elif node.function == "DATE_SUB" and len(node.arguments) == 2:
        left = _convert_node(node.arguments[0], var_types, scope)
        right = _convert_node(node.arguments[1], var_types, scope)
        return f"(- {left} {right})"
    elif node.function == "DATE_ADD" and len(node.arguments) == 2:
        left = _convert_node(node.arguments[0], var_types, scope)
        right = _convert_node(node.arguments[1], var_types, scope)
        return f"(+ {left} {right})"
    elif node.function == "DATEDIFF" and len(node.arguments) == 2:
        left = _convert_node(node.arguments[0], var_types, scope)
        right = _convert_node(node.arguments[1], var_types, scope)
        return f"(- {left} {right})"
    else:
        # Generic uninterpreted function
        if not node.arguments:
            return node.function
        args = " ".join(_convert_node(arg, var_types, scope) for arg in node.arguments)
        return f"({node.function} {args})"


def _convert_literal(node: LiteralNode) -> str:
    if node.data_type == "string":
        # SMT-LIB string literals
        escaped = str(node.value).replace('"', '""')
        return f'"{escaped}"'
    else:
        # Numeric literal
        value = node.value
        if isinstance(value, float) and value == int(value):
            return str(int(value))
        return str(value)


# ---------------------------------------------------------------------------
# Uninterpreted-function declarations
#
# DRC ``FunctionCallNode`` nodes whose head is not one of the
# converter's built-ins (``CURRENT_DATE``, ``DATE_SUB``, ``DATE_ADD``,
# ``DATEDIFF``) are emitted into the SMT script verbatim as
# ``(<head> <args...>)``. cvc5 rejects such calls unless the head has
# been declared, so we must emit a matching ``(declare-fun ...)`` for
# every uninterpreted head, with sorts inferred from the call site.
#
# Two cases the BIRD evaluator hits in practice:
#
# 1. ``LIKE`` — emitted by the SQL→DRC translator from ``col LIKE
#    '%pattern%'``. Always appears at a Boolean position (top-level
#    conjunct in a WHERE clause), with two String operands.
# 2. ``STRFTIME`` — preserved verbatim from BIRD's gold SQL. Appears
#    inside an ordering comparison ``(> (STRFTIME '%Y' dob) '1980')``
#    where both the column and the year literal are coerced to ``Int``
#    by the existing inference rules.
#
# The collector returns one signature per head. When a head shows up
# multiple times in the same condition, we unify the per-position
# sorts the same way relation slots are unified (String wins over
# Int). Return sort follows a similar rule: if any call site demands
# a Bool return (head appears as a child of ``and``/``or``/``not``,
# or as a top-level assertion), we declare the head Bool; otherwise
# the return sort follows whatever sort the surrounding comparison /
# arithmetic context demands.
# ---------------------------------------------------------------------------

# Built-in function heads the SMT converter translates natively, NOT
# as uninterpreted-function calls. These must match the
# ``_convert_function_call`` dispatch above.
_SMT_BUILTIN_FUNCTIONS: frozenset[str] = frozenset(
    {"CURRENT_DATE", "DATE_SUB", "DATE_ADD", "DATEDIFF"}
)


def collect_function_signatures(
    condition: DRCCondition,
    var_types: dict[str, str],
) -> dict[str, tuple[list[str], str]]:
    """Return a ``{function_head: (arg_sorts, return_sort)}`` map for
    every uninterpreted ``FunctionCallNode`` in ``condition``.

    Sort inference:

    * Each argument's sort comes from its expression form — variables
      look up in ``var_types``, literals use their own data type
      (numeric ⇒ Int, string ⇒ String), nested function calls recurse,
      etc. See :func:`_infer_node_sort`.
    * The return sort is determined by where the call appears:
      Bool when it sits at a Boolean position (assertion top, child of
      a logical connective, child of NOT), and the comparison's
      "other side" sort when it sits at a value position. Multiple
      sites unify with String beating Int beating Bool (Bool only
      survives if every site is Bool).
    * Multiple sites for the same head unify per-argument-position
      with the same String-wins rule.

    Built-in heads in :data:`_SMT_BUILTIN_FUNCTIONS` are skipped — they
    are translated natively, not declared.
    """

    signatures: dict[str, tuple[list[str], str]] = {}
    _collect_function_signatures(
        condition, var_types, signatures, return_context="bool",
    )
    return signatures


def _collect_function_signatures(
    node: DRCCondition,
    var_types: dict[str, str],
    sigs: dict[str, tuple[list[str], str]],
    return_context: str,
) -> None:
    """Walk ``node``, recording each uninterpreted function call's
    inferred sort signature. ``return_context`` is the sort the call
    site demands: ``"bool"`` at logical / assertion positions, an
    SMT sort name (``"Int"`` / ``"String"``) at value positions.
    """
    if node is None:
        return

    if isinstance(node, FunctionCallNode):
        if node.function not in _SMT_BUILTIN_FUNCTIONS:
            arg_sorts = [
                _infer_node_sort(a, var_types) for a in node.arguments
            ]
            existing = sigs.get(node.function)
            if existing is None:
                sigs[node.function] = (arg_sorts, return_context)
            else:
                ex_args, ex_ret = existing
                merged_args = _merge_sort_lists(ex_args, arg_sorts)
                merged_ret = _merge_return_sort(ex_ret, return_context)
                sigs[node.function] = (merged_args, merged_ret)
        # Function arguments are value-position. Their sort context is
        # whatever the function declares for that argument position —
        # but since we're inferring that here, we recurse with
        # ``"value"`` (a placeholder meaning "don't tighten the
        # context"); only nested *Boolean* contexts matter for the
        # outer call's return-sort decision.
        for arg in node.arguments:
            _collect_function_signatures(arg, var_types, sigs, "value")
        return

    if isinstance(node, ComparisonNode):
        # Each side is at value position; the other side determines
        # the return-sort context for any function call landing here.
        # Under ordering operators, the same string-coercion rule that
        # ``_convert_node_for_comparison`` uses (date strings → Int,
        # bare-integer strings → Int) applies to the type-inference
        # context too — otherwise a function on the other side gets
        # declared with a String return that mismatches the literal's
        # render as an Int.
        is_ordering = node.operator in ("<", ">", "<=", ">=")
        left_sort = _infer_node_sort_in_context(
            node.right, var_types, is_ordering
        )
        right_sort = _infer_node_sort_in_context(
            node.left, var_types, is_ordering
        )
        _collect_function_signatures(node.left, var_types, sigs, left_sort)
        _collect_function_signatures(node.right, var_types, sigs, right_sort)
        return

    if isinstance(node, ArithmeticNode):
        # Arithmetic operands are Int-context (the SMT layer translates
        # ``+ - * /`` over Int arguments).
        _collect_function_signatures(node.left, var_types, sigs, "Int")
        _collect_function_signatures(node.right, var_types, sigs, "Int")
        return

    if isinstance(node, LogicalConnectiveNode):
        _collect_function_signatures(node.left, var_types, sigs, "bool")
        _collect_function_signatures(node.right, var_types, sigs, "bool")
        return

    if isinstance(node, NotNode):
        _collect_function_signatures(node.operand, var_types, sigs, "bool")
        return

    if isinstance(node, QuantifierNode):
        _collect_function_signatures(node.body, var_types, sigs, "bool")
        return

    # MembershipNode, VariableRefNode, LiteralNode: no nested function
    # calls in our DRC AST, so nothing to recurse into.


def _infer_node_sort(node: DRCCondition, var_types: dict[str, str]) -> str:
    """Best-effort sort inference for an expression-position node.

    Variables look up in ``var_types``, defaulting to ``Int``. Literals
    use their data type. Function calls recurse. Arithmetic is ``Int``.
    Anything else falls back to ``Int`` — the conservative choice that
    matches the rest of the converter's defaults.
    """
    return _infer_node_sort_in_context(node, var_types, is_ordering=False)


def _infer_node_sort_in_context(
    node: DRCCondition, var_types: dict[str, str], is_ordering: bool
) -> str:
    """Like :func:`_infer_node_sort` but aware of ordering-comparison
    coercion.

    Under an ordering operator (``<`` / ``>`` / ``<=`` / ``>=``), a
    string literal that parses as a date or as a bare integer is
    rendered as the corresponding ``Int`` by
    :func:`_convert_node_for_comparison`. The signature inference must
    apply the same rule, otherwise a function on the other side of the
    comparison gets declared with a ``String`` return sort and cvc5
    rejects the comparison with an arithmetic-subterm error.
    """
    if isinstance(node, LiteralNode):
        if node.data_type == "string" and is_ordering:
            if _date_string_to_int(str(node.value)) is not None:
                return "Int"
            if _bare_int_string_to_int(str(node.value)) is not None:
                return "Int"
        if node.data_type == "string":
            return "String"
        return "Int"
    if isinstance(node, VariableRefNode):
        return var_types.get(node.name, "Int")
    if isinstance(node, FunctionCallNode):
        if node.function == "CURRENT_DATE":
            return "Int"
        if node.function in ("DATE_SUB", "DATE_ADD", "DATEDIFF"):
            return "Int"
        return "Int"
    if isinstance(node, ArithmeticNode):
        return "Int"
    return "Int"


def _merge_sort_lists(a: list[str], b: list[str]) -> list[str]:
    """Per-position sort unification with String beating Int beating Bool.

    Lists may have different lengths; the longer one's tail is kept
    verbatim. This matches the relation-slot unification rule in
    :func:`_collect_relation_sorts`.
    """
    out: list[str] = []
    for i in range(max(len(a), len(b))):
        left = a[i] if i < len(a) else None
        right = b[i] if i < len(b) else None
        if left is None:
            out.append(right or "Int")
        elif right is None:
            out.append(left)
        else:
            out.append(_dominant_sort(left, right))
    return out


def _merge_return_sort(a: str, b: str) -> str:
    """Unify two return-sort guesses. ``bool`` only survives if both
    sites demand Bool — otherwise the value-position guess wins.

    We treat ``"value"`` as "no preference"; a more specific guess
    overrides it.
    """
    if a == b:
        return a
    if a == "value":
        return b
    if b == "value":
        return a
    if a == "bool" or b == "bool":
        # One site says Bool, the other a value sort. Value wins —
        # if a function appears in a comparison, it must return a
        # value, even if it also appears bare somewhere (in which
        # case the bare site is genuinely a Bool comparison too,
        # which is unusual but legal).
        return a if b == "bool" else b
    return _dominant_sort(a, b)


def _dominant_sort(a: str, b: str) -> str:
    """Return the dominant sort between ``a`` and ``b``.

    ``String`` beats ``Int`` beats ``Bool``. This mirrors the
    relation-slot unification's "String wins" semantics: if any call
    site uses the function with a String operand or in a String-yielding
    comparison, the declaration must accommodate that or the script
    fails to typecheck.
    """
    rank = {"Bool": 0, "Int": 1, "String": 2}
    if rank.get(a, 1) >= rank.get(b, 1):
        return a
    return b


def _smt_sort(sort_or_context: str) -> str:
    """Translate a return-sort context tag back to an SMT-LIB sort.

    ``"bool"`` (the assertion-position context) maps to ``"Bool"``.
    ``"value"`` (the no-preference fallback) maps to ``"Int"`` —
    nothing in the formula tightened the inference, so default to the
    same sort the rest of the converter falls back to. Anything else
    is already an SMT-LIB sort name and passes through unchanged.
    """
    if sort_or_context == "bool":
        return "Bool"
    if sort_or_context == "value":
        return "Int"
    return sort_or_context
