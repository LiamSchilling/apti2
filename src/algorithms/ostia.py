"""Onward Subsequential Transducer Inference Algorithm (OSTIA).

Type Parameters:
    Q: State type
    U: Input symbol type
    V: Output symbol type
"""
from typing import TypeVar, Callable, Collection, Iterator, Sequence
from util import lcp, ldiv, match, unify
from automata.SFST import SFST
from operations.learner import learn_by_state_merging

Q = TypeVar('Q')
U = TypeVar('U')
V = TypeVar('V')


def ostia(
    input_set: set[U],
    dataset: Collection[tuple[Sequence[U], Sequence[V]]],
    epsilon: Sequence[V],
    concat: Callable[[Sequence[V], Sequence[V]], Sequence[V]],
    choose_transition: Callable[[SFST[Q, U, Sequence[V]], set[Q]], Q],
    search_iter: Callable[[SFST[Q, U, Sequence[V]], set[Q]], Iterator[Q]],
    state_supply: Iterator[Q],
    verbose: bool = False
) -> SFST[Q, U, Sequence[V]]:
    """Learn an SFST from input-output pairs using OSTIA.

    Constructs a minimal subsequential finite-state transducer that maps input sequences
    to output sequences through iterative state merging, with onwardization to ensure
    the result is in a canonical form.

    Args:
        input_set: The alphabet (set of input symbols).
        dataset: Collection of input-output sequence pairs for learning.
        epsilon: The empty/identity sequence.
        concat: Function for concatenating two sequences.
        choose_transition: Heuristic for selecting which frontier transition to process.
        search_iter: Heuristic for iterating through promoted states to try merging with.
        state_supply: Iterator providing fresh state identifiers as needed.
        verbose: Whether to print progress information during learning.

    Returns:
        An SFST that transforms input sequences according to the learned transduction
        from the dataset.
    """
    return learn_by_state_merging(
        input_set=input_set,
        dataset=dataset,
        epsilon=epsilon,
        incr=lambda v: v,
        insertion=lambda v: v,
        contribute=match,
        lmul=concat,
        rmul=concat,
        ldiv=ldiv,
        lcp=lambda vs, epsilon=epsilon: lcp(vs, epsilon),
        try_unify=unify,
        is_epsilon=lambda v: len(v) == 0,
        check_merge=lambda _: True,
        choose_transition=choose_transition,
        search_iter=search_iter,
        state_supply=state_supply,
        postprocess=lambda dfa: dfa,
        verbose=verbose
    )
