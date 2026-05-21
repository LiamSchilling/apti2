"""State merging operations for finite-state transducers.

This module provides the core state merging functionality used in learning algorithms
like RPNI, ALERGIA, OSTIA, and APTI2. It includes recursive merging of individual states
and an iterative framework for repeatedly merging states based on customizable strategies.

Type Parameters:
    Q: State type
    U: Input symbol type
    V: Output value type
    T: Remainder after unification type
"""
from typing import TypeAlias, TypeVar, cast, Callable, Iterator
from copy import copy
from automata.SFST import SFST
from operations.push_output import push_backward

Q = TypeVar('Q')
U = TypeVar('U')
V = TypeVar('V')
T = TypeVar('T')

EdgeKey: TypeAlias = tuple[Q, U] | None
Edge: TypeAlias = tuple[EdgeKey[Q, U], tuple[Q, V]]


def fold_merge(
    fst: SFST[Q, U, V],
    q_src: Q,
    q_dest: Q,
    lmul: Callable[[T, V], V],
    rdiv: Callable[[V, T], V],
    try_unify: Callable[[V, V], tuple[V, T, T] | None],
    is_epsilon: Callable[[T], bool],
) -> bool:
    """Recursively merge state q_src into state q_dest, consolidating transitions and final outputs.

    This function is the core of state merging algorithms. It attempts to unify the behaviors of
    two states by merging their outgoing transitions and final outputs. If unification conflicts
    occur or produce non-epsilon remainders at terminal points, the merge fails and the FST is
    unchanged.

    Args:
        fst: The finite-state transducer being modified.
        q_src: Source state to be merged (will be removed from the FST if merge succeeds).
        q_dest: Destination state where q_src is merged into.
        lmul: Left multiply function that applies a remainder value to the left of an output.
        rdiv: Right divide function that removes a remainder value from the right of an output.
        try_unify: Unification function taking two output values. Returns None on conflict,
                   or a tuple (unified_value, src_remainder, dest_remainder) on success.
        is_epsilon: Predicate checking if a remainder is epsilon (empty/identity element).

    Returns:
        True if the merge succeeds, False if unification conflicts occur or non-epsilon
        remainders are produced at terminal points.

    Behavior:
        1. Merges final outputs: If both states have final outputs, unifies them. Fails if
           unification fails or produces non-epsilon remainders.
        2. Merges transitions: For each input symbol, compares transitions from both states.
           Moves q_src transitions to q_dest; unifies conflicting transitions and recursively
           merges their target states after pushing remainders backward.
    """
    if q_src in fst.final_outputs:
        if q_dest in fst.final_outputs:
            v_src = fst.final_outputs[q_src]
            v_dest = fst.final_outputs[q_dest]
            match try_unify(v_src, v_dest):
                case None:
                    return False
                case (v_uni, src_remainder, dest_remainder):
                    if not is_epsilon(src_remainder) or not is_epsilon(dest_remainder):
                        return False
                    fst.final_outputs[q_dest] = v_uni
        else:
            fst.final_outputs[q_dest] = fst.final_outputs[q_src]

        del fst.final_outputs[q_src]

    for c in fst.input_set:
        if (q_src, c) in fst.transitions:
            if (q_dest, c) in fst.transitions:
                q_src_, v_src = fst.transitions[(q_src, c)]
                q_dest_, v_dest = fst.transitions[(q_dest, c)]
                match try_unify(v_src, v_dest):
                    case None:
                        return False
                    case (v_uni, src_remainder, dest_remainder):
                        push_backward(fst, q_src_, src_remainder, lmul, rdiv)
                        push_backward(fst, q_dest_, dest_remainder, lmul, rdiv)
                        fst.transitions[(q_dest, c)] = q_dest_, v_uni
                        if not fold_merge(fst, q_src_, q_dest_, lmul, rdiv, try_unify, is_epsilon):
                            return False
            else:
                fst.transitions[(q_dest, c)] = fst.transitions[(q_src, c)]

            del fst.transitions[(q_src, c)]

    fst.state_set.remove(q_src)

    return True


def merge(
    fst: SFST[Q, U, V],
    src: Edge[Q, U, V],
    q_dest: Q,
    lmul: Callable[[T, V], V],
    rdiv: Callable[[V, T], V],
    try_unify: Callable[[V, V], tuple[V, T, T] | None],
    is_epsilon: Callable[[T], bool]
) -> bool:
    """Entry point for merging that prepares an edge and delegates to fold_merge.

    The key contribution of merge over fold_merge is to redirect the incoming edge that
    pointed to the source state to instead point to the destination state.

    Args:
        fst: The finite-state transducer being modified.
        src: An Edge representing the source of the merge. Either (None, (q_src_, v)) for
             the initial state transition, or ((q_src, c), (q_src_, v)) for a regular
             transition on input symbol c from state q_src.
        q_dest: Destination state to merge the target of src into.
        lmul: Left multiply function (passed to fold_merge).
        rdiv: Right divide function (passed to fold_merge).
        try_unify: Unification function (passed to fold_merge).
        is_epsilon: Epsilon check predicate (passed to fold_merge).

    Returns:
        True if the merge succeeds (see fold_merge), False otherwise.

    Behavior:
        1. Redirects the incoming edge to point to q_dest instead:
           - If key is None: updates the initial state and initial output to q_dest and v.
           - Otherwise: updates transition (q_src, c) to target q_dest with output v.
        2. Calls fold_merge to perform the recursive merge of q_src_ into q_dest.
    """
    key, (q_src_, v) = src

    match key:
        case None:
            fst.initial_state, fst.initial_output = q_dest, v
        case q_src, c:
            fst.transitions[(q_src, c)] = q_dest, v

    return fold_merge(fst, q_src_, q_dest, lmul, rdiv, try_unify, is_epsilon)


def iterate_merge(
    fst: SFST[Q, U, V],
    try_merge: Callable[[SFST[Q, U, V], Edge[Q, U, V], Q], bool],
    choose_transition: Callable[[SFST[Q, U, V], set[Edge[Q, U, V]]], Edge[Q, U, V]],
    search_iter: Callable[[SFST[Q, U, V], set[Q]], Iterator[Q]]
) -> SFST[Q, U, V]:
    """Iteratively merge states in an FST using customizable selection and search strategies.

    This forms the outer loop of state merging algorithms like RPNI, ALERGIA, OSTIA, and APTI2.

    Args:
        fst: The finite-state transducer to merge.
        try_merge: Function that attempts to merge the target state of the given transition into the
                   given state. That is, on a successful merge, the joined state should have the
                   name of the given *state*, the *second* argument. Returns True if merge succeeds.
        choose_transition: Function that selects a transition from the frontier set.
        search_iter: Function that yields candidate states from the promoted set to try merging
                     with.

    Returns:
        A new SFST with states merged. The original FST is not modified.
    """
    promoted = cast(set[Q], set())
    frontier = {(cast(EdgeKey[Q, U], None), (fst.initial_state, fst.initial_output))}

    while frontier != set():
        src = choose_transition(fst, frontier)
        _, (q_src, _) = src

        success = False
        for q_dest in search_iter(fst, promoted):
            fst_ = copy(fst)
            if try_merge(fst_, src, q_dest):
                fst = fst_
                success = True
                break

        if not success:
            promoted.add(q_src)

        frontier = {
            (cast(EdgeKey[Q, U], (q, c)), (q_, v))
            for q in promoted
            for c, q_, v in fst.iter_outgoing_states_from(q)
            if q_ not in promoted
        }

    return fst
