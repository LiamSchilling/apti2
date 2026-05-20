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
from SFST import SFST

Q = TypeVar('Q')
U = TypeVar('U')
V = TypeVar('V')
T = TypeVar('T')


def push_forward(
    fst: SFST[Q, U, V],
    q: Q,
    pref: T,
    rmul: Callable[[V, T], V],
    ldiv: Callable[[T, V], V]
) -> None:
    """Push a prefix forward from a state through the FST.

    This operation normalizes output distribution at state q by moving a common prefix
    from outgoing edges to incoming edges. The prefix is multiplied onto transitions
    entering q and left-divided from transitions leaving q, preserving the transduction.

    Args:
        fst: The SFST to modify (modified in-place).
        q: The state at which to apply push-forward.
        pref: The prefix to push forward (typically the LCP of outgoing outputs).
        mul: Binary multiplication operation to combine prefix with outputs.
        ldiv: Binary left division to remove prefix from outputs.
    """
    if q == fst.initial_state:
        fst.initial_output = rmul(fst.initial_output, pref)

    for key, (q_, v) in fst.transitions.items():
        if q == q_:
            fst.transitions[key] = q_, rmul(v, pref)

    for c in fst.input_set:
        if (q, c) in fst.transitions:
            q_, v = fst.transitions[(q, c)]
            fst.transitions[(q, c)] = q_, ldiv(pref, v)

    if q in fst.final_outputs:
        v = fst.final_outputs[q]
        fst.final_outputs[q] = ldiv(pref, v)


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
