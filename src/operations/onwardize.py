"""Onwardization of finite state transducers.

This module implements the onwardization algorithm, which transforms subsequential
finite state transducers (SFSTs) into a canonical form by pushing outputs forward
through the state machine.

Type Parameters:
    Q: State type
    U: Input symbol type
    V: Output value type
    T: Result of LCP type
"""
from typing import TypeVar, Callable
from FSTs.SFST import SFST
from push_output import push_forward

Q = TypeVar('Q')
U = TypeVar('U')
V = TypeVar('V')
T = TypeVar('T')


def onwardize_trim_acyclic(
    fst: SFST[Q, U, V],
    rmul: Callable[[V, T], V],
    ldiv: Callable[[T, V], V],
    lcp: Callable[[set[V]], T]
) -> None:
    """Onwardize a trim, acyclic SFST by pushing outputs forward through states.

    This function modifies the SFST in-place by computing a common prefix of local outputs
    at each state and pushing that prefix forward to transitions entering the state.
    The states must be processed in a reverse topological order.

    Args:
        fst: The trim, acyclic SFST to onwardize (modified in-place).
        mul: Binary multiplication operation for outputs (e.g., concatenation).
             Combines prefix with existing output: mul(prefix, output).
        ldiv: Binary left division operation for outputs (e.g., string difference).
              Removes prefix from output: ldiv(prefix, output).
        lcp: Function to compute the longest common prefix from a set of outputs.
             Returns the longest prefix common to all outputs in the set.
    """
    for q in fst.iter_accessible_states_from(fst.initial_state, set()):
        pref = lcp(set(fst.iter_outgoing_from(q)))
        push_forward(fst, q, pref, rmul, ldiv)
