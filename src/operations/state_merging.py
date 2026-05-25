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
from typing import TypeVar, Callable, Iterator
from copy import copy
from automata.SFST import SFST
from operations.push_output import push_outgoing

Q = TypeVar('Q')
U = TypeVar('U')
V = TypeVar('V')
T = TypeVar('T')


def fold_merge(
    fst: SFST[Q, U, V],
    q_src: Q,
    q_dest: Q,
    lmul: Callable[[T, V], V],
    try_unify: Callable[[V, V], tuple[V, T] | None],
    is_epsilon: Callable[[T], bool],
    verbose : bool = False
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
        try_unify: Unification function taking two output values (v_src, v_dest). Returns a tuple
                   (unified_value, src_remainder). Returns None on failure.
        is_epsilon: Predicate checking if a remainder is epsilon (empty/identity element).
        verbose: Flag for printing debugging information.

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
    if verbose:
        print(f"attempting to merge state {q_src} into state {q_dest}")

    if q_src in fst.final_outputs:
        v_src = fst.final_outputs[q_src]
        if q_dest in fst.final_outputs:
            v_dest = fst.final_outputs[q_dest]
            match try_unify(v_src, v_dest):
                case None:
                    return False
                case v_uni, src_remainder:
                    if not is_epsilon(src_remainder):
                        return False
                    fst.final_outputs[q_dest] = v_uni
        else:
            fst.final_outputs[q_dest] = v_src

        del fst.final_outputs[q_src]

    for c in fst.input_set:
        if (q_src, c) in fst.transitions:
            q_src_, v_src = fst.transitions[(q_src, c)]
            if (q_dest, c) in fst.transitions:
                q_dest_, v_dest = fst.transitions[(q_dest, c)]
                match try_unify(v_src, v_dest):
                    case None:
                        return False
                    case v_uni, src_remainder:
                        push_outgoing(fst, q_src_, src_remainder, lmul)
                        fst.transitions[(q_dest, c)] = q_dest_, v_uni
                        if not fold_merge(
                            fst,
                            q_src_,
                            q_dest_,
                            lmul,
                            try_unify,
                            is_epsilon,
                            verbose=verbose
                        ):
                            return False
            else:
                if verbose:
                    print(f"directing state {q_dest} on input {c} to state {q_src_}")
                fst.transitions[(q_dest, c)] = q_src_, v_src

            del fst.transitions[(q_src, c)]

    fst.state_set.remove(q_src)

    return True


def merge(
    fst: SFST[Q, U, V],
    q_src: Q,
    q_dest: Q,
    tr_src_incoming: tuple[Q, U] | None,
    lmul: Callable[[T, V], V],
    try_unify: Callable[[V, V], tuple[V, T] | None],
    is_epsilon: Callable[[T], bool],
    verbose : bool = False
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
        try_unify: Unification function (passed to fold_merge).
        is_epsilon: Epsilon check predicate (passed to fold_merge).
        verbose: Flag for printing debugging information.

    Returns:
        True if the merge succeeds (see fold_merge), False otherwise.

    Behavior:
        1. Redirects the incoming edge to point to q_dest instead:
           - If key is None: updates the initial state and initial output to q_dest and v.
           - Otherwise: updates transition (q_src, c) to target q_dest with output v.
        2. Calls fold_merge to perform the recursive merge of q_src_ into q_dest.
    """
    match tr_src_incoming:
        case None:
            if verbose:
                print(f"redirecting initial transition to state {q_dest}")
            fst.initial_state = q_dest
        case q_src_origin, c:
            if verbose:
                print(f"redirecting state {q_src_origin} on input {c} to state {q_dest}")
            _, v = fst.transitions[(q_src_origin, c)]
            fst.transitions[(q_src_origin, c)] = q_dest, v

    success = fold_merge(
        fst,
        q_src,
        q_dest,
        lmul,
        try_unify,
        is_epsilon,
        verbose=verbose
    )

    if success and verbose:
        print("finished merges, maybe pending final validation")

    return success


def iterate_merge(
    fst: SFST[Q, U, V],
    try_merge: Callable[[SFST[Q, U, V], Q, Q, tuple[Q, U] | None], bool],
    choose_transition: Callable[[SFST[Q, U, V], set[Q]], Q],
    search_iter: Callable[[SFST[Q, U, V], set[Q]], Iterator[Q]],
    verbose : bool = False
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
        verbose: Flag for printing debugging information.

    Returns:
        A new SFST with states merged. The original FST is not modified.
    """
    promoted: set[Q] = set()
    frontier: dict[Q, tuple[Q, U] | None] = {fst.initial_state : None}

    while frontier != {}:
        q_src = choose_transition(fst, set(frontier))

        success = False
        for q_dest in search_iter(fst, promoted):
            fst_ = copy(fst)
            if verbose:
                print(f"merge candidate: {q_src} into {q_dest}")
            if try_merge(fst_, q_src, q_dest, frontier[q_src]):
                if verbose:
                    print("success\n")
                fst = fst_
                success = True
                break
            elif verbose:
                print("failure\n")

        if not success:
            promoted.add(q_src)

        frontier: dict[Q, tuple[Q, U] | None] = {
            q_ : (q, c)
            for q in promoted
            for c, q_, _ in fst.iter_outgoing_states_from(q)
            if q_ not in promoted
        }

    return fst
