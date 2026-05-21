"""Push output operations for finite-state transducers.

This module provides functions to normalize output distributions in FSTs by
pushing outputs forward or backward through states. These operations redistribute
output values while preserving the transduction semantics, useful for weight
normalization and lattice optimization.

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


def push_outputs(
    fst: SFST[Q, U, V],
    q: Q,
    elem: T,
    rop: Callable[[V, T], V],
    lop: Callable[[T, V], V]
) -> None:
    """Redistribute an element through a state by updating all incident transitions.

    Modifies the FST by pushing elem through state q:
    - Applies right-multiplication (rop) to transitions entering q and initial output
    - Applies left-multiplication (lop) to transitions leaving q and final output

    Args:
        fst: The SFST to modify (modified in-place).
        q: The state through which to push the element.
        elem: The element to push through (typically an output semiring value).
        rop: Right operation to apply to incoming transitions: rop(output, elem).
        lop: Left operation to apply to outgoing transitions: lop(elem, output).
    """
    if q == fst.initial_state:
        fst.initial_output = rop(fst.initial_output, elem)

    for key, (q_, v) in fst.transitions.items():
        if q == q_:
            fst.transitions[key] = q_, rop(v, elem)

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
        mul: Binary multiplication operation to combine prefix with outputs.
        ldiv: Binary left division to remove prefix from outputs.
    """
    push_outputs(fst, q, pref, rmul, ldiv)


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
    push_outputs(fst, q, suff, rdiv, lmul)
