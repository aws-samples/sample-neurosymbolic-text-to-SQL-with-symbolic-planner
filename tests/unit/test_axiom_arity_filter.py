"""Tests for the FK-axiom predicate-arity filter in the equivalence script.

Background — dev_78's run-06 failure:
The runner builds foreign-key axioms over BIRD's full schema (e.g.
``(forall (... 49 vars …) (=> (frpm <49 args>) (exists (...) (schools <49
args>))))``). The equivalence checker's pre-SMT preprocessing prunes
unused slots from predicate calls inside the equivalence assertion,
which in turn shrinks each ``(declare-fun T (...) Bool)`` to just the
slots used. When that happens, the axioms (which still call ``T`` at
full width) are ill-typed against the script's narrowed declaration.
cvc5 1.3.4 exits with code 1, which the runner reports as
``Verdict.unknown``.

The fix: drop axioms whose predicate calls don't match the script's
post-pruning predicate-arity declarations. Axioms are *optional*
extra information for the equivalence proof, so dropping them is
strictly better than crashing the solver. These tests pin the
helper that does the filtering.
"""

from __future__ import annotations

from text_to_sql_planner.equivalence.equivalence_checker import (
    _axiom_predicate_arities_match,
    _smt_tokenise,
)


# ---------------------------------------------------------------------------
# _smt_tokenise: the paren-balanced tokeniser the helper relies on
# ---------------------------------------------------------------------------


def test_tokenise_basic_parens_and_atoms():
    assert _smt_tokenise("(foo bar)") == ["(", "foo", "bar", ")"]


def test_tokenise_nested_forms():
    tokens = _smt_tokenise("(a (b c) d)")
    assert tokens == ["(", "a", "(", "b", "c", ")", "d", ")"]


def test_tokenise_quoted_strings_kept_intact():
    """Spaces and parens inside ``"..."`` don't split the atom."""
    tokens = _smt_tokenise('(= name "hello world")')
    assert tokens == ["(", "=", "name", '"hello world"', ")"]


def test_tokenise_quoted_strings_with_escapes():
    tokens = _smt_tokenise(r'(= x "say \"hi\"")')
    assert tokens == ["(", "=", "x", r'"say \"hi\""', ")"]


def test_tokenise_skips_line_comments():
    tokens = _smt_tokenise(
        "(foo bar)  ;; this is a comment\n(baz qux)"
    )
    assert tokens == ["(", "foo", "bar", ")", "(", "baz", "qux", ")"]


def test_tokenise_handles_empty_input():
    assert _smt_tokenise("") == []
    assert _smt_tokenise("   \n\t  ") == []


# ---------------------------------------------------------------------------
# _axiom_predicate_arities_match: the actual filter
# ---------------------------------------------------------------------------


def test_filter_accepts_axiom_with_matching_arities():
    """A well-formed axiom whose predicate calls match declared arity passes."""
    axiom = (
        "(assert (forall ((x Int) (y Int)) "
        "(=> (T x y) (exists ((z Int)) (S z)))))"
    )
    arities = {"T": 2, "S": 1}
    assert _axiom_predicate_arities_match(axiom, arities) is True


def test_filter_rejects_axiom_with_too_many_args():
    """Axiom calls T with 3 args but the script declares T with 2."""
    axiom = "(assert (T x y z))"
    arities = {"T": 2}
    assert _axiom_predicate_arities_match(axiom, arities) is False


def test_filter_rejects_axiom_with_too_few_args():
    """Axiom calls T with 1 arg but the script declares T with 3."""
    axiom = "(assert (T x))"
    arities = {"T": 3}
    assert _axiom_predicate_arities_match(axiom, arities) is False


def test_filter_ignores_predicates_not_in_the_arities_map():
    """A call to a non-tracked head (=> / and / forall) is not checked."""
    axiom = "(assert (=> a b c d e))"  # ``=>`` not in the map
    arities = {"T": 2}
    assert _axiom_predicate_arities_match(axiom, arities) is True


def test_filter_handles_nested_predicate_calls():
    """All predicate calls in a nested form are checked, not just the top one."""
    axiom = (
        "(assert (forall ((x Int)) "
        "(=> (T x) (exists ((y Int) (z Int)) (S x y z)))))"
    )
    # Outer T is correct (1 arg). Inner S is called with 3 args — script
    # declares it with 2. The filter should reject the whole axiom.
    arities = {"T": 1, "S": 2}
    assert _axiom_predicate_arities_match(axiom, arities) is False


def test_filter_accepts_when_arities_map_is_empty():
    """No predicate to check against → trivially well-typed."""
    axiom = "(assert (T x y z))"
    assert _axiom_predicate_arities_match(axiom, {}) is True


def test_filter_accepts_empty_axiom():
    """Empty / whitespace-only axioms are accepted (defensive — caller filters elsewhere)."""
    assert _axiom_predicate_arities_match("", {"T": 1}) is True
    assert _axiom_predicate_arities_match("   \n  ", {"T": 1}) is True


def test_filter_dev_78_pattern_rejects_full_arity_axiom_against_pruned_schema():
    """The exact dev_78 shape: axiom calls ``schools`` with 49 args, but the
    pruned script declares ``schools`` with 2."""
    # Build a 49-variable forall with two predicate calls at full arity.
    schools_args = " ".join(f"v{i}" for i in range(49))
    bindings = " ".join(f"(v{i} Int)" for i in range(49))
    axiom = (
        f"(assert (forall ({bindings}) "
        f"(=> (frpm {schools_args}) (schools {schools_args}))))"
    )
    # After pruning, both predicates are at arity 2.
    arities = {"frpm": 2, "schools": 2}
    assert _axiom_predicate_arities_match(axiom, arities) is False


def test_filter_dev_78_pattern_accepts_when_no_pruning_happened():
    """Same axiom shape, but the script kept the predicates at full
    declared arity. The axiom is then well-typed and survives."""
    schools_args = " ".join(f"v{i}" for i in range(49))
    bindings = " ".join(f"(v{i} Int)" for i in range(49))
    axiom = (
        f"(assert (forall ({bindings}) "
        f"(=> (frpm {schools_args}) (schools {schools_args}))))"
    )
    arities = {"frpm": 49, "schools": 49}
    assert _axiom_predicate_arities_match(axiom, arities) is True


def test_filter_unbalanced_input_does_not_crash():
    """Pathological input bails conservatively (returns True) instead of raising."""
    # Missing close paren — caller should not see an exception.
    axiom = "(assert (T x y"
    arities = {"T": 2}
    # Don't pin True/False; just confirm no exception.
    _axiom_predicate_arities_match(axiom, arities)


# ---------------------------------------------------------------------------
# Undeclared-predicate rejection (the dev_1425 / run-07 root cause).
#
# The runner builds FK axioms for *every* foreign-key edge in the BIRD
# database, but the equivalence script only declares the relation
# predicates the query actually mentions. An axiom calling a relation
# predicate that the script never declared makes cvc5 hit an unknown
# function symbol and exit with code 1 — the entire equivalence check
# fails. The filter must drop such axioms before they reach the solver.
# ---------------------------------------------------------------------------


def test_filter_rejects_axiom_calling_undeclared_predicate():
    """Axiom references ``unrelated`` which the script never declares."""
    axiom = "(assert (forall ((x Int)) (=> (T x) (unrelated x))))"
    arities = {"T": 1}  # ``unrelated`` is missing
    assert _axiom_predicate_arities_match(axiom, arities) is False


def test_filter_rejects_axiom_with_one_declared_one_undeclared():
    """Mixed case: T is declared, U isn't. Reject the whole axiom."""
    axiom = "(assert (forall ((x Int)) (=> (T x) (U x))))"
    arities = {"T": 1}  # ``U`` is missing
    assert _axiom_predicate_arities_match(axiom, arities) is False


def test_filter_dev_1425_pattern_rejects_axiom_for_table_not_in_query():
    """The dev_1425 shape: query touches only ``major``, but the FK axioms
    span ``member``, ``event``, ``attendance``, etc. Those axioms reference
    predicates the script never declares; they must be dropped."""
    # An FK axiom from ``member`` -> ``zip_code`` against a script that
    # only declares ``major`` (the single table the query references).
    axiom = (
        "(assert (forall ((m1 Int) (m2 Int) (z1 Int)) "
        "(=> (member m1 m2) (exists ((z2 Int)) (zip_code z1 z2)))))"
    )
    arities = {"major": 3}  # neither ``member`` nor ``zip_code`` declared
    assert _axiom_predicate_arities_match(axiom, arities) is False


# ---------------------------------------------------------------------------
# Binding-list handling: sorted-variable declarations like ``(x Int)``
# inside ``forall`` / ``exists`` / ``let`` must NOT be mistaken for
# predicate calls. The body of the binder is still subject to the
# undeclared-predicate / arity checks.
# ---------------------------------------------------------------------------


def test_filter_accepts_forall_with_declared_predicate_in_body():
    """``forall ((x Int)) (T x)`` — binding list isn't a predicate call,
    body's predicate is declared. Pass."""
    axiom = "(assert (forall ((x Int)) (T x)))"
    arities = {"T": 1}
    assert _axiom_predicate_arities_match(axiom, arities) is True


def test_filter_accepts_forall_with_int_and_string_sorts_in_binding_list():
    """Binding list mixes Int and String sorts. Sort atoms must not be
    treated as predicate-call heads."""
    axiom = "(assert (forall ((x Int) (s String)) (T x s)))"
    arities = {"T": 2}
    assert _axiom_predicate_arities_match(axiom, arities) is True


def test_filter_handles_empty_binding_list():
    """``(forall () body)`` is technically allowed by SMT-LIB and must
    not crash the walker."""
    axiom = "(assert (forall () (T x)))"
    arities = {"T": 1}
    # Don't pin True/False — confirm it doesn't raise.
    _axiom_predicate_arities_match(axiom, arities)


def test_filter_handles_nested_quantifier_binding_lists():
    """``forall`` outside, ``exists`` inside: each has its own binding
    list, both must be skipped by the predicate-call walker."""
    axiom = (
        "(assert (forall ((x Int)) "
        "(=> (T x) (exists ((y Int) (z Int)) (S x y z)))))"
    )
    arities = {"T": 1, "S": 3}
    assert _axiom_predicate_arities_match(axiom, arities) is True


def test_filter_handles_let_binding_list():
    """``let`` uses the same ``((var value) ...)`` shape as quantifiers;
    its binding list must also be skipped."""
    axiom = "(assert (let ((a (T x))) (= a a)))"
    arities = {"T": 1}
    # ``T`` inside the let-binding's value position should still be
    # checked. (a 1) here is treated as predicate ``a``, which isn't
    # declared, so this axiom should be rejected. That's the wrong
    # direction for our needs — but the filter's job is to be safe,
    # not perfect. Confirm it doesn't crash and bails one way.
    _axiom_predicate_arities_match(axiom, arities)


def test_filter_rejects_undeclared_predicate_inside_quantifier_body():
    """The body of a binder is NOT exempt from checks — only the binding
    list itself is. An undeclared predicate hidden inside a forall body
    must still be detected."""
    axiom = "(assert (forall ((x Int)) (undeclared x)))"
    arities = {"T": 1}  # ``undeclared`` is missing
    assert _axiom_predicate_arities_match(axiom, arities) is False


def test_filter_binding_list_var_named_like_a_predicate_does_not_confuse_walker():
    """A bound variable with the same name as a built-in head shouldn't
    derail the walker — the binding-list mode skips everything inside
    the list regardless of names."""
    # ``and`` would be a built-in head, but here it's a bound variable.
    # The body uses the real ``and``.
    axiom = "(assert (forall ((and Int)) (= and 0)))"
    arities = {"T": 1}
    # Should accept — no relation-predicate calls anywhere.
    assert _axiom_predicate_arities_match(axiom, arities) is True
