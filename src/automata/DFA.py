"""Deterministic Finite Automaton implementation.

This module specializes SFST to deterministic finite automata (DFAs), where each
transition output is None. Acceptance is determined by state membership in final_outputs.

Type Parameters:
    Q: State type
    U: Input symbol type
"""
from typing import TypeVar, TypeAlias
from automata.SFST import SFST, assert_SFST

Q = TypeVar('Q')
U = TypeVar('U')


DFA: TypeAlias = SFST[Q, U, None]
"""Deterministic Finite Automaton.

A specialization of SFST where output values are None. Acceptance is determined by
whether a state appears in final_outputs.

Type Parameters:
    Q: State type.
    U: Input symbol type.

Invariants:
    The same as the SFST invariants.

Example:
    >>> # Create a simple DFA accepting strings ending with 'a'
    >>> # Type parameters: Q=int, U=str
    >>> dfa = DFA(
    ...     state_set={0, 1},
    ...     input_set={'a', 'b'},
    ...     initial_state=0,
    ...     transitions={
    ...         (0, 'a'): (1, None),
    ...         (0, 'b'): (0, None),
    ...         (1, 'a'): (1, None),
    ...         (1, 'b'): (0, None),
    ...     },
    ...     initial_output=None,
    ...     final_outputs={1: None}
    ... )
"""


def assert_DFA(dfa: DFA[Q, U]):
    """Validate that a DFA satisfies DFA invariants.

    Args:
        dfa: A DFA[Q, U] value to validate.

    Raises:
        AssertionError: If the DFA violates DFA invariants.
    """
    assert_SFST(dfa)
