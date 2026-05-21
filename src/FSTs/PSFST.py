"""Probabilistic Subsequential Finite State Transducer implementation.

This module specializes SFST to probability-annotated transducers, where each
transition output is paired with a real-valued probability.

Type Parameters:
    Q: State type
    U: Input symbol type
    V: Output value type
    A: Accumulator type
"""
from typing import TypeAlias, TypeVar, Callable, Mapping
from SFST import SFST, assert_SFST

Q = TypeVar('Q')
U = TypeVar('U')
V = TypeVar('V')
A = TypeVar('A')


WithProb: TypeAlias = tuple[V, float]
"""Probability-annotated value type.

Represents a value paired with a real-valued probability.

Type Parameter:
    V: The underlying value type.

Invariants:
    - The probability component must be in the range [0.0, 1.0].
"""


def assert_WithProb(vp: WithProb[V]) -> None:
    """Validate that a WithProb value satisfies probability invariants.

    Args:
        vp: A WithProb[V] value (value, probability) pair to validate.

    Raises:
        AssertionError: If the probability component is outside [0.0, 1.0].
    """
    _, p = vp
    assert 0 <= p and p <= 1, \
        f"probability {p} is not in the range [0.0, 1.0]"


PSFST = SFST[Q, U, WithProb[V]]
"""Probabilistic Sequential Finite State Transducer.

A specialization of SFST where output values use the WithProb type annotation,
pairing each output value with a real-valued probability.

Type Parameters:
    Q: State type.
    U: Input symbol type.
    V: The underlying output value type.

Invariants (in addition to SFST invariants):
    - The probability component of `initial_output` must equal 1.0.
    - For every state `q` in `state_set`, the sum of probability components of all
      outgoing transitions `(q, u) -> q'` in `transitions` for all `u` and `q'`,
      plus the probability component of `final_outputs[q]`, must equal 1.0.

Example:
    >>> # Create a simple probabilistic language model
    >>> # Type parameters: Q=int, U=str, V=str
    >>> psfst = PSFST(
    ...     state_set={0, 1},
    ...     input_set={'a', 'b'},
    ...     initial_state=0,
    ...     transitions={
    ...         (0, 'a'): (1, ('emit_a', 0.5)),
    ...         (0, 'b'): (1, ('emit_b', 0.5)),
    ...         (1, 'a'): (1, ('loop_a', 0.3)),
    ...         (1, 'b'): (1, ('loop_b', 0.3)),
    ...     },
    ...     initial_output=('start', 1.0),
    ...     final_outputs={0: ('end_0', 0.0), 1: ('end_1', 0.4)}
    ... )
"""


def outgoing_probabilities(fst: PSFST[Q, U, V]) -> Mapping[Q, float]:
    """Compute the total outgoing probability from all states, which should be 1.0 for every state.

    Sums the probability components of all transitions from state `q` and
    the probability component of the final output for `q` (if defined).

    Args:
        fst: The PSFST to analyze.

    Returns:
        A mapping from each state in `state_set` to its total outgoing probability.
    """
    probs = {q: 0.0 for q in fst.state_set}
    for q, (_, p) in fst.iter_outgoing():
        probs[q] += p
    return probs


def assert_PSFST(fst: PSFST[Q, U, V], delta: float) -> None:
    """Validate that a PSFST instance satisfies all invariants.

    Args:
        fst: The PSFST to validate.
        delta: Tolerance for probability sum validation. Outgoing probability
               from each state must be within [1.0 - delta, 1.0 + delta].

    Raises:
        AssertionError: If any SFST invariant is violated, if any output's
                        probability is outside [0.0, 1.0], if the initial
                        output probability is not 1.0, or if any state's
                        total outgoing probability is not within delta of 1.0.
    """
    assert_SFST(fst)

    for vp in fst.iter_outputs():
        assert_WithProb(vp)

    _, p = fst.initial_output
    assert 1.0 - delta <= p and p <= 1.0 + delta, \
        f"`initial_output` probability {p} is not 1.0"

    outgoing_probs = outgoing_probabilities(fst)
    for q in fst.state_set:
        p = outgoing_probs[q]
        assert 1.0 - delta <= p and p <= 1.0 + delta, \
            f"bad state {q} with outgoing probability {p}, not 1.0"


def lift_acc_to_prob(acc: Callable[[A, V], A]) -> Callable[[WithProb[A], WithProb[V]], WithProb[A]]:
    """Lift an accumulator function to handle WithProb values.

    Adapts an accumulator that works with bare values to work with probability-
    annotated values. Applies the accumulator to the value components while
    multiplying the probability components.

    Args:
        acc: An accumulator function `(A, V) -> A` operating on bare values.

    Returns:
        A lifted accumulator `(WithProb[A], WithProb[V]) -> WithProb[A]` that
        applies `acc` to the value components and multiplies the probabilities.

    Example:
        >>> psfst = PSFST(...)  # Probabilistic transducer with outputs like ("symbol", 0.5)
        >>> concat_acc = lambda acc, v: acc + v  # Concatenate string outputs
        >>> lifted = lift_acc_to_prob(concat_acc)
        >>> result = run(psfst, ["a", "b"], ("", 1.0), lifted)
        >>> # result is (concatenated_string, path_probability)
    """
    return lambda ap, vp: (acc(ap[0], vp[0]), ap[1] * vp[1])
