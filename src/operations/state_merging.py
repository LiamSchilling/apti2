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
from typing import TypeVar, Any, Callable, Iterator, Reversible
from automata.SFST import SFST
from operations.push_output import push_outgoing

Q = TypeVar('Q')
U = TypeVar('U')
V = TypeVar('V')
T = TypeVar('T')


def _run_all_rev(ops: Reversible[Callable[[], Any]]) -> None:
    """Execute all functions in an iterable in reverse order.

    This is a helper function for running sequences of side-effect operations,
    particularly useful for executing accumulated undo operations in reverse order.

    Args:
        ops: An iterable of nullary callables to execute. Each function is called
             with no arguments.
    """
    for op in reversed(ops):
        op()


def fold_merge(
    fst: SFST[Q, U, V],
    q_src: Q,
    q_dest: Q,
    lmul: Callable[[T, V], V],
    ldiv: Callable[[T, V], V],
    try_unify: Callable[[V, V], tuple[V, T] | None],
    is_epsilon: Callable[[T], bool],
    verbose : bool = False
) -> Callable[[], None] | None:
    """Recursively merge state q_src into state q_dest, consolidating transitions and final outputs.

    This function is the core of state merging algorithms. It attempts to unify the behaviors of
    two states by merging their outgoing transitions and final outputs. All modifications to the
    FST are accumulated into an undo function, which is returned on success. If unification
    conflicts occur or produce non-epsilon remainders at terminal points, the merge fails, the
    undo function is called to restore the FST, and None is returned.

    Args:
        fst: The finite-state transducer being modified.
        q_src: Source state to be merged (will be removed from the FST if merge succeeds).
        q_dest: Destination state where q_src is merged into.
        lmul: Left multiply function that applies a remainder value to the left of an output.
        ldiv: Left divide function that reverses a remainder value application (used for undo).
        try_unify: Unification function taking two output values (v_src, v_dest). Returns a tuple
                   (unified_value, src_remainder). Returns None on failure.
        is_epsilon: Predicate checking if a remainder is epsilon (empty/identity element).
        verbose: Flag for printing debugging information.

    Returns:
        On success: A callable that undoes all modifications made by this merge (and any nested
                    fold_merge calls). Calling the undo function restores the FST to its state
                    before this merge attempt.
        On failure: None, indicating the merge failed and the FST has been restored to its
                    pre-merge state.

    Behavior:
        1. Merges final outputs: If both states have final outputs, unifies them. Fails if
           unification fails or produces non-epsilon remainders.
        2. Merges transitions: For each input symbol, compares transitions from both states.
           Moves q_src transitions to q_dest; unifies conflicting transitions and recursively
           merges their target states after pushing remainders backward.
        3. All modifications are collected in a list and executed in reverse order on undo,
           avoiding deep recursion.
        4. On failure, the accumulated undo operations are executed in reverse to restore the FST.
    """
    undo_ops: list[Callable[[], None]] = []

    if verbose:
        print(f"attempting to merge state {q_src} into state {q_dest}")

    if q_src in fst.final_outputs:
        v_src = fst.final_outputs[q_src]
        if q_dest in fst.final_outputs:
            v_dest = fst.final_outputs[q_dest]
            match try_unify(v_src, v_dest):
                case None:
                    return _run_all_rev(undo_ops)
                case v_uni, src_remainder:
                    if not is_epsilon(src_remainder):
                        return _run_all_rev(undo_ops)
                    fst.final_outputs[q_dest] = v_uni
                    undo_ops.append(lambda fst=fst, q=q_dest, v=v_dest: fst.final_outputs.update({q: v}))
        else:
            fst.final_outputs[q_dest] = v_src
            undo_ops.append(lambda fst=fst, q=q_dest: (fst.final_outputs.pop(q, None), None)[1])

        del fst.final_outputs[q_src]
        undo_ops.append(lambda fst=fst, q=q_src, v=v_src: fst.final_outputs.update({q: v}))

    for c in fst.input_set:
        if (q_src, c) in fst.transitions:
            q_src_, v_src = fst.transitions[(q_src, c)]
            if (q_dest, c) in fst.transitions:
                q_dest_, v_dest = fst.transitions[(q_dest, c)]
                match try_unify(v_src, v_dest):
                    case None:
                        return _run_all_rev(undo_ops)
                    case v_uni, src_remainder:
                        push_outgoing(fst, q_src_, src_remainder, lmul)
                        undo_ops.append(lambda fst=fst, ldiv=ldiv, q_=q_src_, r=src_remainder: push_outgoing(fst, q_, r, ldiv))
                        fst.transitions[(q_dest, c)] = q_dest_, v_uni
                        undo_ops.append(lambda fst=fst, q=q_dest, c=c, q_=q_dest_, v=v_dest: fst.transitions.update({(q, c): (q_, v)}))
                        match fold_merge(
                            fst,
                            q_src_,
                            q_dest_,
                            lmul,
                            ldiv,
                            try_unify,
                            is_epsilon,
                            verbose=verbose
                        ):
                            case None:
                                return _run_all_rev(undo_ops)
                            case undo_fold_merge:
                                undo_ops.append(undo_fold_merge)
            else:
                if verbose:
                    print(f"directing state {q_dest} on input {c} to state {q_src_}")
                fst.transitions[(q_dest, c)] = q_src_, v_src
                undo_ops.append(lambda fst=fst, q=q_dest, c=c: (fst.transitions.pop((q, c), None), None)[1])

            del fst.transitions[(q_src, c)]
            undo_ops.append(lambda fst=fst, q=q_src, c=c, q_=q_src_, v=v_src: fst.transitions.update({(q, c): (q_, v)}))

    fst.state_set.remove(q_src)
    undo_ops.append(lambda fst=fst, q=q_src: fst.state_set.add(q))

    return lambda undo_ops=undo_ops: _run_all_rev(undo_ops)


def merge(
    fst: SFST[Q, U, V],
    q_src: Q,
    q_dest: Q,
    tr_src_incoming: tuple[Q, U] | None,
    lmul: Callable[[T, V], V],
    ldiv: Callable[[T, V], V],
    try_unify: Callable[[V, V], tuple[V, T] | None],
    is_epsilon: Callable[[T], bool],
    verbose : bool = False
) -> Callable[[], None] | None:
    """Entry point for merging that prepares an incoming edge and delegates to fold_merge.

    This function extends fold_merge by redirecting the incoming edge that pointed to the
    source state to instead point to the destination state. Like fold_merge, it accumulates
    undo operations and returns a combined undo function on success or None on failure.

    Args:
        fst: The finite-state transducer being modified.
        q_src: Source state to be merged.
        q_dest: Destination state to merge q_src into.
        tr_src_incoming: The incoming transition to redirect. Either None for the initial
                         state, or (q_src_origin, c) for a transition from q_src_origin
                         on input symbol c.
        lmul: Left multiply function (passed to fold_merge).
        ldiv: Left divide function (passed to fold_merge).
        try_unify: Unification function (passed to fold_merge).
        is_epsilon: Epsilon check predicate (passed to fold_merge).
        verbose: Flag for printing debugging information.

    Returns:
        On success: A callable that undoes all modifications from both the incoming edge
                    redirection and the recursive fold_merge operation.
        On failure: None, and all modifications (including the edge redirection) have been
                    undone via the accumulated undo function.

    Behavior:
        1. Redirects the incoming edge to point to q_dest:
           - If tr_src_incoming is None: updates the initial state to q_dest.
           - Otherwise: updates transition (q_src_origin, c) to target q_dest instead of q_src.
        2. Calls fold_merge to recursively merge q_src into q_dest.
        3. If fold_merge succeeds, returns the combined undo function.
        4. If fold_merge fails, all accumulated undo operations are executed to restore both the
           redirection and any partial merge changes.
    """
    undo_ops: list[Callable[[], None]] = []

    match tr_src_incoming:
        case None:
            if verbose:
                print(f"redirecting initial transition to state {q_dest}")
            fst.initial_state = q_dest
            undo_ops.append(lambda fst=fst, q_=q_src: setattr(fst, 'initial_state', q_))
        case q_src_origin, c:
            if verbose:
                print(f"redirecting state {q_src_origin} on input {c} to state {q_dest}")
            _, v = fst.transitions[(q_src_origin, c)]
            fst.transitions[(q_src_origin, c)] = q_dest, v
            undo_ops.append(lambda fst=fst, q=q_src_origin, c=c, q_=q_src, v=v: fst.transitions.update({(q, c): (q_, v)}))

    match fold_merge(
        fst,
        q_src,
        q_dest,
        lmul,
        ldiv,
        try_unify,
        is_epsilon,
        verbose=verbose
    ):
        case None:
            if verbose:
                print("merges failed")
            return _run_all_rev(undo_ops)
        case undo_fold_merge:
            if verbose:
                print("finished merges, maybe pending final validation")
            undo_ops.append(undo_fold_merge)
            return lambda undo_ops=undo_ops: _run_all_rev(undo_ops)


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
            if verbose:
                print(f"merge candidate: {q_src} into {q_dest}")
            if try_merge(fst, q_src, q_dest, frontier[q_src]):
                if verbose:
                    print("success\n")
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
