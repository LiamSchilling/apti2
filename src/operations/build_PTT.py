"""Prefix Tree Transducers (PTT) implementation.

This module provides functions to build prefix tree transducers,
which are finite state transducers organized as tries. PTTs accumulate data values
along input paths. The implementation allows flexible data accumulation through three customizable
operations: incrementation on revisits, insertion on first occurrence, and
contribution when merging values at shared final states.

Type Parameters:
    Q: State type
    U: Input symbol type
    V: Output value type
    D: Output component of data pair type
"""
from typing import TypeVar, Callable, Collection, Iterator, Sequence
from FSTs.SFST import SFST

Q = TypeVar('Q')
U = TypeVar('U')
V = TypeVar('V')
D = TypeVar('D')


def insert_data_PTT(
    fst: SFST[Q, U, V],
    data: tuple[Sequence[U], D],
    epsilon: V,
    incr: Callable[[V], V],
    insertion: Callable[[D], V],
    contribute: Callable[[V, D], V],
    state_supply: Iterator[Q]
) -> None:
    """Insert a single (input sequence, data output value) pair into a prefix tree transducer.

    This function modifies the transducer in-place by creating or following a path
    through the trie corresponding to the input sequence, accumulating the data value
    at the terminal state using customizable operations.

    Args:
        fst: The prefix tree transducer to modify (modified in-place).
        data: Tuple of (input_sequence, data_value) to insert.
        epsilon: The identity/empty value for the output type V (used to initialize new transitions).
        incr: Unary operation applied to outputs when a path is revisited.
              Used to track path frequency. Signature: V -> V
        insertion: Unary operation converting a data value into an output on first insertion.
                   Used to wrap/initialize the data. Signature: D -> V
        contribute: Binary operation to merge new data into existing output at a final state.
                    Used to accumulate values. Signature: (V, D) -> V
        state_supply: Iterator that generates fresh state identifiers for new trie nodes.
    """
    u, d = data

    fst.initial_output = incr(fst.initial_output)

    q = fst.initial_state
    for c in u:
        if (q, c) in fst.transitions:
            q_, v = fst.transitions[(q, c)]
        else:
            q_, v = next(state_supply), epsilon
            fst.state_set.add(q_)
        fst.transitions[(q, c)] = q_, incr(v)
        q = q_

    if q in fst.final_outputs:
        v = fst.final_outputs[q]
        fst.final_outputs[q] = contribute(v, d)
    else:
        fst.final_outputs[q] = insertion(d)


def build_PTT(
    input_set: set[U],
    dataset: Collection[tuple[Sequence[U], D]],
    epsilon: V,
    incr: Callable[[V], V],
    insertion: Callable[[D], V],
    contribute: Callable[[V, D], V],
    state_supply: Iterator[Q]
) -> SFST[Q, U, V]:
    """Build a prefix tree transducer from a collection of (sequence, data) pairs.

    Constructs a trie-structured transducer that stores data values at terminal states.
    All input sequences in the dataset create paths through the transducer, with
    shared prefixes reusing states. Data values are accumulated at final states
    using the customizable accumulation operations.

    Args:
        input_set: The alphabet of input symbols that the PTT accepts.
        dataset: Collection of (input_sequence, data_value) pairs to insert into the trie.
                 Sequences with common prefixes will share trie paths.
        epsilon: The identity/empty value for output type V (initializes new transitions).
        incr: Unary operation applied when paths are revisited (e.g., increment counter).
              Signature: V -> V
        insertion: Unary operation to wrap/initialize data values on first insertion.
                   Signature: D -> V
        contribute: Binary operation to accumulate data values at shared final states.
                    Signature: (V, D) -> V
        state_supply: Iterator that generates fresh state identifiers for new trie nodes.

    Returns:
        A new SFST[Q, U, V] representing the prefix tree structure with accumulated values.
    """
    q0 = next(state_supply)
    fst = SFST(
        state_set={q0},
        input_set=input_set,
        initial_state=q0,
        transitions={},
        initial_output=epsilon,
        final_outputs={}
    )

    for data in dataset:
        insert_data_PTT(fst, data, epsilon, incr, insertion, contribute, state_supply)

    return fst
