"""Subsequential Finite State Transducer implementation.

This module provides a generic framework for symbolic finite state transducers,
which are automata that transform input sequences into output sequences by
transitioning through states and accumulating outputs.

Type Parameters:
    Q: State type
    U: Input symbol type
    V: Output value type
    A: Accumulator type
"""
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Iterator, MutableMapping, Sequence

Q = TypeVar('Q')
U = TypeVar('U')
V = TypeVar('V')
A = TypeVar('A')


@dataclass
class SFST(Generic[Q, U, V]):
    """A Subsequential Finite State Transducer.

    An SFST is a state machine that produces outputs as it transitions through
    states based on input symbols. It supports initial outputs, transition-based outputs,
    and final state-based outputs. Partiality is also supported.

    Type Parameters:
        Q: State type.
        U: Input symbol type.
        V: Output value type.

    Attributes:
        state_set: Set of all possible states in the automaton.
        input_set: Set of all possible input symbols.
        initial_state: The starting state for execution.
        transitions: Mapping from (state, input) pairs to (next_state, output) tuples.
        initial_output: Output produced before processing the first input.
        final_outputs: Mapping from final states to output values.

    Invariants:
        - All states refered to in the machine are elements of `state_set`.
        - All inputs refered to in the machine are elements of `input_set`.

    Example:
        >>> # Create a simple transducer that counts inputs
        >>> # Type parameters: Q=int, U=str, V=int
        >>> fst = SFST(
        ...     state_set={0, 1, 2},
        ...     input_set={'a', 'b'},
        ...     initial_state=0,
        ...     transitions={
        ...         (0, 'a'): (1, 1),
        ...         (0, 'b'): (2, 1),
        ...         (1, 'a'): (1, 1),
        ...         (1, 'b'): (2, 1),
        ...     },
        ...     initial_output=0,
        ...     final_outputs={0: 0, 1: 0, 2: 0}
        ... )
    """
    state_set: set[Q]
    input_set: set[U]
    initial_state: Q
    transitions: MutableMapping[tuple[Q, U], tuple[Q, V]]
    initial_output: V
    final_outputs: MutableMapping[Q, V]

    def iter_outputs(self) -> Iterator[V]:
        """Iterate over all outputs in the machine.

        Yields all output values V from the transducer, including:
        - The initial output
        - All outputs from transitions
        - All outputs from final states

        Yields:
            Each output value V in the machine.
        """
        yield self.initial_output
        for _, v in self.transitions.values():
            yield v
        for v in self.final_outputs.values():
            yield v

    def iter_ingoing(self) -> Iterator[tuple[Q, V]]:
        """Iterate over all incoming edges (states with their outputs).

        Yields all (state, output) pairs representing incoming edges to states
        in the machine, including the initial state and all next states from
        transitions.

        Yields:
            Tuples of (state, output) for each incoming edge.
        """
        yield self.initial_state, self.initial_output
        for q_, v in self.transitions.values():
            yield q_, v

    def iter_outgoing(self) -> Iterator[tuple[Q, V]]:
        """Iterate over all outgoing edges (states with their outputs).

        Yields all (state, output) pairs representing outgoing edges from states
        in the machine, including source states from transitions and final states
        with their outputs.

        Yields:
            Tuples of (state, output) for each outgoing edge.
        """
        for (q, _), (_, v) in self.transitions.items():
            yield q, v
        for q, v in self.final_outputs.items():
            yield q, v

    def iter_outgoing_from(self, q: Q) -> Iterator[V]:
        """Iterate over all output values from outgoing edges of a given state.

        Yields all output values associated with transitions leaving the given state,
        as well as the final output of the state itself (if it exists).

        Args:
            q: The state from which to find outgoing outputs.

        Yields:
            Each output value V from transitions leaving the state and the final output of the state.
        """
        for c in self.input_set:
            if (q, c) in self.transitions:
                _, v = self.transitions[(q, c)]
                yield v
        if q in self.final_outputs:
            yield self.final_outputs[q]

    def iter_accessible_states_from(self, q: Q, ignore: set[Q]) -> Iterator[Q]:
        """Iterate over all states in the machine accessible from some start state.

        Performs a depth-first search over the machine,
        and yields a state only once all of its children have been processed.
        For acyclic machines, this guarantees that states are yielded in a reverse
        topological order.

        Args:
            q: The starting state from which to find accessible states.
            ignore: A set of states to ignore during traversal; used to track visited states
                    and prevent infinite loops when processing cyclic paths.

        Yields:
            Each state in the machine that is accessible from the given state.
        """
        if q in ignore:
            return
        ignore.add(q)

        for c in self.input_set:
            if (q, c) in self.transitions:
                q_, _ = self.transitions[(q, c)]
                yield from self.iter_accessible_states_from(q_, ignore)

        ignore.remove(q)
        yield q


def assert_SFST(fst: SFST[Q, U, V]) -> None:
    """Validate that an SFST instance satisfies all invariants documented in the SFST class.

    Args:
        fst: The SFST to validate.

    Raises:
        AssertionError: If any invariant is violated, with a message describing
                        the violation.
    """
    assert fst.initial_state in fst.state_set, \
        f"`initial_state` {fst.initial_state} is not in `state_set`"

    for (q, u), (q_, _) in fst.transitions.items():
        assert q in fst.state_set, \
            f"source state {q} in `transitions` is not in `state_set`"
        assert u in fst.input_set, \
            f"input {u} in `transitions` is not in `input_set`"
        assert q_ in fst.state_set, \
            f"next state {q_} in `transitions` is not in `state_set`"

    for q, _ in fst.final_outputs.items():
        assert q in fst.state_set, \
            f"state {q} in `final_outputs` is not in `state_set`"


def run_from(
    fst: SFST[Q, U, V],
    q: Q,
    u: Sequence[U],
    init: A,
    acc: Callable[[A, V], A]
) -> tuple[Q, A] | None:
    """Execute the transducer from a given state.

    Process an input sequence starting from a specified state, accumulating
    outputs from transitions only. This function does NOT include `initial_output`
    or `final_outputs` in the accumulation; only the outputs from the transitions
    themselves are accumulated.

    Args:
        fst: The SFST to execute.
        q: The starting state.
        u: Input sequence to process.
        init: Initial accumulator value.
        acc: Accumulator function taking (accumulator, output) and returning
             the updated accumulator.

    Returns:
        A tuple of (final_state, accumulated_result) after processing all inputs,
        or `None` if a required transition is not defined.

    Example:
        >>> fst = SFST(...)  # Some transducer
        >>> result = run_from(fst, 0, ['a', 'b'], 0, lambda acc, v: acc + v)
        >>> if result is not None:
        ...     final_state, count = result
    """
    a = init
    for c in u:
        if (q, c) in fst.transitions:
            q, v = fst.transitions[(q, c)]
            a = acc(a, v)
        else:
            return None
    return q, a


def run(
    fst: SFST[Q, U, V],
    u: Sequence[U],
    init: A,
    acc: Callable[[A, V], A]
) -> tuple[Q, A] | None:
    """Execute the transducer from its initial state.

    Process an input sequence starting from the transducer's initial state,
    including the initial output and final output accumulation.

    Args:
        fst: The SFST to execute.
        u: Input sequence to process.
        init: Initial accumulator value.
        acc: Accumulator function taking (accumulator, output) and returning
             the updated accumulator.

    Returns:
        A tuple of (final_state, accumulated_result) after processing all inputs
        and applying final state outputs, or `None` if any required transition or
        final output is not defined.

    Example:
        >>> fst = SFST(...)  # Some transducer
        >>> result = run(fst, ['a', 'b'], 0, lambda acc, v: acc + v)
        >>> if result is not None:
        ...     final_state, total = result
    """
    a = acc(init, fst.initial_output)
    match run_from(fst, fst.initial_state, u, a, acc):
        case None:
            return None
        case q, a:
            if q in fst.final_outputs:
                return q, acc(a, fst.final_outputs[q])
            else:
                return None
