"""Regular Positive and Negative Inference (RPNI) algorithm.

Type Parameters:
    Q: State type
    U: Input symbol type
"""
from typing import TypeVar, Callable, Collection, Iterator, Sequence
from automata.DFA import DFA
from automata.SFST import run
from operations.learner import Edge, learn_by_state_merging

Q = TypeVar('Q')
U = TypeVar('U')


def check_merge(dfa: DFA[Q, U], neg_dataset: Collection[Sequence[U]]) -> bool:
    """Verify that a DFA rejects all negative examples.

    Checks whether the given DFA correctly rejects every sequence in the negative dataset
    by running each sequence through the DFA and confirming the output is None.

    Args:
        dfa: The DFA to validate.
        neg_dataset: Collection of input sequences that should be rejected.

    Returns:
        True if the DFA rejects all negative examples, False otherwise.
    """
    return all(run(dfa, u, None, lambda none, _ : none) == None for u in neg_dataset)


def rpni(
    input_set: set[U],
    pos_dataset: Collection[Sequence[U]],
    neg_dataset: Collection[Sequence[U]],
    choose_transition: Callable[[DFA[Q, U], set[Edge[Q, U, None]]], Edge[Q, U, None]],
    search_iter: Callable[[DFA[Q, U], set[Q]], Iterator[Q]],
    state_supply: Iterator[Q],
    verbose: bool = False
) -> DFA[Q, U]:
    """Learn a DFA from positive and negative examples using RPNI.

    Constructs a minimal DFA that accepts all strings in pos_dataset and rejects all
    strings in neg_dataset through iterative state merging.

    Args:
        input_set: The alphabet (set of input symbols).
        pos_dataset: Collection of input sequences to accept (positive examples).
        neg_dataset: Collection of input sequences to reject (negative examples).
        choose_transition: Heuristic for selecting which frontier transition to process.
        search_iter: Heuristic for iterating through promoted states to try merging with.
        state_supply: Iterator providing fresh state identifiers as needed.
        verbose: Whether to print progress information during learning.

    Returns:
        A DFA that accepts all positive examples and rejects all negative examples.
    """
    if verbose:
        print(
            f"with the following negative data: [\n\t" +
            "\n\t".join(f"{u}, {None}" for u in neg_dataset) +
            "\n]\n"
        )

    return learn_by_state_merging(
        input_set=input_set,
        dataset=[(u, None) for u in pos_dataset],
        epsilon=None,
        incr=lambda none : none,
        insertion=lambda none : none,
        contribute=lambda none, _ : none,
        lmul=lambda _, none : none,
        rmul=lambda none, _ : none,
        ldiv=lambda _, none : none,
        rdiv=lambda none, _ : none,
        lcp=lambda _ : None,
        try_unify=lambda none, _ : (none, None, None),
        is_epsilon=lambda _ : True,
        check_merge=lambda dfa : check_merge(dfa, neg_dataset),
        choose_transition=choose_transition,
        search_iter=search_iter,
        state_supply=state_supply,
        postprocess=lambda dfa : dfa,
        verbose=verbose
    )
