"""Push output operations for finite-state transducers.

This module provides functions to normalize output distributions in FSTs by
pushing outputs forward or backward through states. These operations redistribute
output values while preserving the transduction semantics.

Type Parameters:
    Q: State type
    U: Input symbol type
    V: Output value type
    T: Result of LCP type
"""
from typing import TypeVar, Callable
from automata.SFST import SFST

Q = TypeVar('Q')
U = TypeVar('U')
V = TypeVar('V')
T = TypeVar('T')


def push_ingoing(
    fst: SFST[Q, U, V],
    q: Q,
    elem: T,
    rop: Callable[[V, T], V]
) -> None:
    """Apply an operation to all incoming transitions to a state.

    Modifies the output values on all transitions that end at state q by applying
    the right operation. If q is the initial state, also modifies the initial output.

    Args:
        fst: The SFST to modify (modified in-place).
        q: The target state.
        elem: The element to apply to incoming transitions.
        rop: Binary right operation to combine with transition outputs.
    """
    if q == fst.initial_state:
        fst.initial_output = rop(fst.initial_output, elem)

    for key, (q_, v) in fst.transitions.items():
        if q == q_:
            fst.transitions[key] = q_, rop(v, elem)


def push_outgoing(
    fst: SFST[Q, U, V],
    q: Q,
    elem: T,
    lop: Callable[[T, V], V]
) -> None:
    """Apply an operation to all outgoing transitions from a state.

    Modifies the output values on all transitions that start from state q by applying
    the left operation. If q is a final state, also modifies its final output.

    Args:
        fst: The SFST to modify (modified in-place).
        q: The source state.
        elem: The element to apply to outgoing transitions.
        lop: Binary left operation to combine with transition outputs.
    """
    for c, q_, v in fst.iter_outgoing_states_from(q):
        fst.transitions[(q, c)] = q_, lop(elem, v)

    if q in fst.final_outputs:
        v = fst.final_outputs[q]
        fst.final_outputs[q] = lop(elem, v)


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
        rmul: Binary right multiplication operation to combine prefix with outputs.
        ldiv: Binary left division to remove prefix from outputs.
    """
    push_ingoing(fst, q, pref, rmul)
    push_outgoing(fst, q, pref, ldiv)


def push_backward(
    fst: SFST[Q, U, V],
    q: Q,
    suff: T,
    lmul: Callable[[T, V], V],
    rdiv: Callable[[V, T], V]
) -> None:
    """Push a suffix backward from a state through the FST.

    This operation normalizes output distribution at state q by moving a common suffix
    from incoming edges to outgoing edges. The suffix is left-multiplied onto transitions
    leaving q and right-divided from transitions entering q, preserving the transduction.

    Args:
        fst: The SFST to modify (modified in-place).
        q: The state at which to apply push-backward.
        suff: The suffix to push backward (typically the LCP of incoming outputs).
        lmul: Binary left multiplication to combine suffix with outputs.
        rdiv: Binary right division to remove suffix from outputs.
    """
    push_outgoing(fst, q, suff, lmul)
    push_ingoing(fst, q, suff, rdiv)
