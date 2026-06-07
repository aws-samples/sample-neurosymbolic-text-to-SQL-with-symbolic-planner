"""Regression test for per-position sort unification across DRC sites.

The dev_1519 BIRD failure was caused by ``_collect_relation_sorts``
picking the *longest* membership-arg list and dropping shorter sites'
type observations. When the generated DRC bound ``Date`` and ``Time``
to string literals at a 4-arg membership and the gold DRC's full-row
9-arg membership had every slot resolved to ``Int`` (because none of
its locally-bound variables were near a string literal), the longer
list won and the merged predicate signature came out
``(Int Int Int Int Int Int Int Int Int)``. cvc5 then rejected the
script with a sort error when the 4-arg side passed ``String``
constants into ``Int`` slots.

The fix: unify per-position with String-wins-over-Int and length =
max-arity. This file pins that behaviour directly on the helper.
"""

from __future__ import annotations

from text_to_sql_planner.equivalence.smt_converter import (
    _collect_relation_sorts,
)
from text_to_sql_planner.types.drc import (
    LogicalConnectiveNode,
    MembershipNode,
)


def _and(*conds):
    """Build a left-folded conjunction tree."""
    if not conds:
        raise ValueError("need at least one operand")
    if len(conds) == 1:
        return conds[0]
    head, *tail = conds
    return LogicalConnectiveNode(operator="and", left=head, right=_and(*tail))


def test_per_position_string_wins_when_short_site_observes_string():
    """A 4-arg membership with String at position 0 unifies into a 9-arg signature.

    Reproduces the dev_1519 shape: the generated side's ``transactions_1k``
    membership has Date / Time / GasStationID / Price as four args, of
    which positions 0 and 1 are typed ``String``. The gold side's
    full-row membership has nine args, all typed ``Int`` because none
    of its locally-bound variables had a string-literal comparison.
    The unified signature must be 9 long and have ``String`` at
    positions 0 and 1.
    """

    membership_short = MembershipNode(
        variables=["Date", "Time", "GasStationID", "Price"],
        relation="transactions_1k",
    )
    membership_long = MembershipNode(
        variables=[
            "v_tid", "v_date", "v_time", "v_cust",
            "v_card", "v_gid", "v_pid", "v_amt", "v_price",
        ],
        relation="transactions_1k",
    )

    var_types = {
        # Generated-side bindings: Date / Time observed as String.
        "Date": "String",
        "Time": "String",
        "GasStationID": "Int",
        "Price": "Int",
        # Gold-side bindings: locally inferred as Int because none of
        # them were near a string literal.
        "v_tid": "Int",
        "v_date": "Int",
        "v_time": "Int",
        "v_cust": "Int",
        "v_card": "Int",
        "v_gid": "Int",
        "v_pid": "Int",
        "v_amt": "Int",
        "v_price": "Int",
    }
    rel_sorts: dict[str, list[str]] = {}

    # Site order matters for the bug: the long list comes first so the
    # short list has a chance to overwrite it. The fix has to make the
    # outcome insensitive to ordering.
    _collect_relation_sorts(
        _and(membership_long, membership_short), var_types, rel_sorts
    )

    sig = rel_sorts["transactions_1k"]
    assert len(sig) == 9, "signature length should be the max arity"
    assert sig[0] == "String", "position 0 was observed as String at the short site"
    assert sig[1] == "String", "position 1 was observed as String at the short site"
    # Positions only observed at the long site keep their Int sort.
    for i in range(2, 9):
        assert sig[i] == "Int", f"position {i} was only seen as Int"


def test_per_position_unification_is_order_independent():
    """The unified signature must not depend on which site is visited first."""

    short = MembershipNode(
        variables=["a", "b"],
        relation="t",
    )
    long_ = MembershipNode(
        variables=["x", "y", "z"],
        relation="t",
    )

    var_types_first = {"a": "String", "b": "Int", "x": "Int", "y": "Int", "z": "Int"}

    rel_sorts_a: dict[str, list[str]] = {}
    _collect_relation_sorts(_and(short, long_), var_types_first, rel_sorts_a)

    rel_sorts_b: dict[str, list[str]] = {}
    _collect_relation_sorts(_and(long_, short), var_types_first, rel_sorts_b)

    assert rel_sorts_a == rel_sorts_b
    # And both arrived at the right answer.
    assert rel_sorts_a["t"] == ["String", "Int", "Int"]


def test_long_site_string_propagates_to_unified_signature():
    """When only the LONG site observes String, the unified signature still gets it."""

    short = MembershipNode(variables=["a", "b"], relation="t")
    long_ = MembershipNode(variables=["x", "y", "z"], relation="t")

    var_types = {"a": "Int", "b": "Int", "x": "Int", "y": "Int", "z": "String"}
    rel_sorts: dict[str, list[str]] = {}
    _collect_relation_sorts(_and(short, long_), var_types, rel_sorts)

    # Position 2 was only observed at the long site, where it's String.
    assert rel_sorts["t"] == ["Int", "Int", "String"]


def test_single_membership_still_records_correct_signature():
    """Single-site case: the unification helper must not regress the trivial path."""

    membership = MembershipNode(variables=["a", "b", "c"], relation="t")
    var_types = {"a": "Int", "b": "String", "c": "Int"}
    rel_sorts: dict[str, list[str]] = {}
    _collect_relation_sorts(membership, var_types, rel_sorts)

    assert rel_sorts == {"t": ["Int", "String", "Int"]}


def test_empty_observations_default_to_int():
    """Variables with no observed type fall back to Int — the SMT default."""

    short = MembershipNode(variables=["a", "b"], relation="t")
    long_ = MembershipNode(variables=["x", "y", "z"], relation="t")
    # No types observed for any variable.
    var_types: dict[str, str] = {}
    rel_sorts: dict[str, list[str]] = {}
    _collect_relation_sorts(_and(short, long_), var_types, rel_sorts)

    assert rel_sorts == {"t": ["Int", "Int", "Int"]}
