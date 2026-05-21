"""Frequency Subsequential Finite State Transducer implementation.

This module specializes SFST to frequency-annotated transducers, where each
transition output is paired with an integer frequency count.

Type Parameters:
    Q: State type
    U: Input symbol type
    V: Output value type
    A: Accumulator type
"""
from typing import TypeAlias, TypeVar, Callable, Mapping
from automata.SFST import SFST, assert_SFST

Q = TypeVar('Q')
U = TypeVar('U')
V = TypeVar('V')
A = TypeVar('A')


WithFreq: TypeAlias = tuple[V, int]
"""Frequency-annotated value type.

Represents a value paired with an integer frequency count.

Type Parameter:
    V: The underlying value type.

Invariants:
    - The frequency component must be non-negative (>= 0).
"""


def assert_WithFreq(vf: WithFreq[V]) -> None:
    """Validate that a WithFreq value satisfies frequency invariants.

    Args:
        vf: A WithFreq[V] value (value, frequency) pair to validate.

    Raises:
        AssertionError: If the frequency component is negative.
    """
    _, f = vf
    assert f >= 0, \
        f"frequency {f} is negative"


FSFST: TypeAlias = SFST[Q, U, WithFreq[V]]
"""Frequency Sequential Finite State Transducer.

A specialization of SFST where output values use the WithFreq type annotation,
pairing each output value with an integer frequency count.

Type Parameters:
    Q: State type.
    U: Input symbol type.
    V: The underlying output value type.

Invariants (in addition to SFST invariants):
    - For every state `q` in `state_set`, the total ingoing frequency equals the
      total outgoing frequency (flow conservation).
    - Ingoing frequency for a state `q` is defined as:
      - The frequency component of `initial_output` if `q` is `initial_state`.
      - Plus the sum of frequency components of all transitions `(q', u) -> q`
        in `transitions` for all `q'` and `u`.
    - Outgoing frequency for a state `q` is defined as:
      - The sum of frequency components of all transitions `(q, u) -> q'`
        in `transitions` for all `u` and `q'`.
      - Plus the frequency component of `final_outputs[q]`.

Example:
    >>> # Create a frequency transducer
    >>> # Type parameters: Q=int, U=str, V=str
    >>> fsfst = FSFST(
    ...     state_set={0, 1},
    ...     input_set={'a', 'b'},
    ...     initial_state=0,
    ...     transitions={
    ...         (0, 'a'): (1, ('output_a', 3)),
    ...         (0, 'b'): (1, ('output_b', 3)),
    ...         (1, 'a'): (1, ('loop', 6)),
    ...     },
    ...     initial_output=('start', 6),
    ...     final_outputs={1: ('end', 0)}
    ... )
"""


def ingoing_frequencies(fst: FSFST[Q, U, V]) -> Mapping[Q, int]:
    """Compute the total ingoing frequency for all states.

    For each state, sums:
    - The frequency component of `initial_output` (for the initial state)
    - The frequency components of all transitions `(q', u) -> q` ending at that state

    Args:
        fst: The FSFST to analyze.

    Returns:
        A mapping from each state in `state_set` to its total ingoing frequency.
    """
    freqs = {q: 0 for q in fst.state_set}
    for q, (_, f) in fst.iter_ingoing():
        freqs[q] += f
    return freqs


def outgoing_frequencies(fst: FSFST[Q, U, V]) -> Mapping[Q, int]:
    """Compute the total outgoing frequency from all states.

    Sums the frequency components of all transitions from state `q` and
    the frequency component of the final output for `q` (if defined).

    Args:
        fst: The FSFST to analyze.

    Returns:
        A mapping from each state in `state_set` to its total outgoing frequency.
    """
    freqs = {q: 0 for q in fst.state_set}
    for q, (_, f) in fst.iter_outgoing():
        freqs[q] += f
    return freqs


def assert_FSFST(fst: FSFST[Q, U, V]) -> None:
    """Validate that an FSFST instance satisfies all invariants.

    Args:
        fst: The FSFST to validate.

    Raises:
        AssertionError: If any SFST invariant is violated, if any output's
                        frequency is negative, or if flow conservation fails
                        at any state.
    """
    assert_SFST(fst)

    for vf in fst.iter_outputs():
        assert_WithFreq(vf)

    ingoing_freqs = ingoing_frequencies(fst)
    outgoing_freqs = outgoing_frequencies(fst)
    for q in fst.state_set:
        assert ingoing_freqs[q] == outgoing_freqs[q], (
            f"flow conservation failed at state {q} with "
            f"ingoing frequency {ingoing_freqs[q]} and "
            f"outgoing frequency {outgoing_freqs[q]}"
        )


def lift_acc_to_freq(acc: Callable[[A, V], A]) -> Callable[[A, WithFreq[V]], A]:
    """Lift an accumulator function to handle WithFreq values.

    Adapts an accumulator that works with bare values to work with frequency-
    annotated values by extracting the value component and discarding the frequency.

    Args:
        acc: An accumulator function `(A, V) -> A` operating on bare values.

    Returns:
        A lifted accumulator `(A, WithFreq[V]) -> A` that extracts the value
        component from the frequency-annotated output and applies `acc`.

    Example:
        >>> fsfst = FSFST(...)  # Frequency transducer with outputs like ("a", 5)
        >>> concat_acc = lambda acc, v: acc + v  # Concatenate string outputs
        >>> lifted = lift_acc_to_freq(concat_acc)
        >>> result = run(fsfst, ["a", "b"], "", lifted)
        >>> # result extracts string values, ignoring frequencies
    """
    return lambda a, vf: acc(a, vf[0])
