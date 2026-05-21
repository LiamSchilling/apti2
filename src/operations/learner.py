"""Learning finite-state transducers by state merging.

This module implements the core learning algorithm that constructs finite-state transducers
from sample data through state merging. The process involves three main stages:
1. Building a prefix tree transducer (PTT) from the dataset
2. Onwardizing the PTT to a canonical form
3. Iteratively merging states based on provided compatibility operations

Type Parameters:
    Q: State type
    U: Input symbol type
    V: Output value type
    V_: Final output value type (may differ from V after application of the continuation)
    D: Output component of data pair type
    T: Result of LCP type
"""
from typing import TypeVar, Callable, Collection, Iterator, Sequence
from automata.SFST import SFST
from operations.build_PTT import build_PTT
from operations.onwardize import onwardize_trim_acyclic
from operations.state_merging import Edge, merge, iterate_merge

Q = TypeVar('Q')
U = TypeVar('U')
V = TypeVar('V')
V_ = TypeVar('V_')
D = TypeVar('D')
T = TypeVar('T')


def learn_by_state_merging(
    input_set: set[U],
    dataset: Collection[tuple[Sequence[U], D]],
    epsilon: V,
    incr: Callable[[V], V],
    insertion: Callable[[D], V],
    contribute: Callable[[V, D], V],
    lmul: Callable[[T, V], V],
    rmul: Callable[[V, T], V],
    ldiv: Callable[[T, V], V],
    rdiv: Callable[[V, T], V],
    lcp: Callable[[set[V]], T],
    try_unify: Callable[[V, V], tuple[V, T, T] | None],
    is_epsilon: Callable[[T], bool],
    check_merge: Callable[[SFST[Q, U, V]], bool],
    choose_transition: Callable[[SFST[Q, U, V], set[Edge[Q, U, V]]], Edge[Q, U, V]],
    search_iter: Callable[[SFST[Q, U, V], set[Q]], Iterator[Q]],
    state_supply: Iterator[Q],
    postprocess: Callable[[SFST[Q, U, V]], SFST[Q, U, V_]],
    verbose: bool = False
) -> SFST[Q, U, V_]:
    """Learn an SFST from training data through prefix-tree construction and state merging.

    This is the main entry point for learning FSTs. The algorithm follows the ALERGIA/APTI2
    paradigm: build a prefix tree from the data, normalize it, then iteratively merge
    compatible states based on provided search and merge strategies.

    Args:
        input_set: The set of possible input symbols.
        dataset: Collection of (input_sequence, output_data) pairs to learn from.
        epsilon: Identity element for output values (acts as the empty output).
        incr: Increment function called when revisiting a transition with new data.
        insertion: Function converting output data to output values on first visit.
        contribute: Function combining an accumulated value with new data at final states.
        lmul: Left multiply - applies remainder to left of output value.
        rmul: Right multiply - applies remainder to right of output value.
        ldiv: Left divide - removes remainder from left of output value.
        rdiv: Right divide - removes remainder from right of output value.
        lcp: Longest common prefix function for outputs.
        try_unify: Attempts to unify two output values, returning unified value and
                   remainder suffixes, or None on conflict.
        is_epsilon: Predicate to check if a value is epsilon.
        check_merge: Validation function that checks if a tentative merge should be accepted.
                     Called after merge succeeds; returns True to accept, False to reject.
        choose_transition: Heuristic selecting which transition to process from the frontier.
        search_iter: Heuristic yielding candidate states to try merging with.
        state_supply: Iterator providing fresh state identifiers as needed.
        postprocess: Function to transform final transducer (e.g., normalize outputs).
        verbose: Whether to print progress information during learning.

    Returns:
        A learned SFST of type SFST[Q, U, V_].

    Algorithm:
        1. Builds a prefix tree transducer from the dataset
        2. Onwardizes it to canonical form with outputs pushed forward
        3. Iteratively merges compatible states
        4. Applies an output-modifying continuation to the final transducer
    """
    if verbose:
        print(
            f"learning from the following positive data by state-merging: [\n\t" +
            "\n\t".join(f"{u}, {d}" for u, d in dataset) +
            "\n]\n"
        )

    fst = build_PTT(input_set, dataset, epsilon, incr, insertion, contribute, state_supply)

    if verbose:
        print(f"naively constructed PTT:\n{fst}\n")

    onwardize_trim_acyclic(fst, rmul, ldiv, lcp)

    if verbose:
        print(f"onwardized PTT:\n{fst}\n")

    def try_merge(fst_: SFST[Q, U, V], src: Edge[Q, U, V], q_dest: Q) -> bool:
        if not merge(fst_, src, q_dest, lmul, rdiv, try_unify, is_epsilon):
            return False
        return check_merge(fst_)

    fst = iterate_merge(fst, try_merge, choose_transition, search_iter)

    if verbose:
        print(f"FST after state-merging:\n{fst}\n")

    fst = postprocess(fst)

    if verbose:
        print(f"result FST of learning algorithm:\n{fst}")

    return fst
