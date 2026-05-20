"""Conversion from Frequency SFST to Probabilistic SFST.

This module provides functionality to convert a frequency-annotated finite state
transducer (FSFST) to a probabilistically-annotated finite state transducer
(PSFST) by normalizing frequency counts to probabilities.

Type Parameters:
    Q: State type
    U: Input symbol type
    V: Output value type
"""
from typing import TypeVar, cast
from FSFST import FSFST, outgoing_frequencies
from PSFST import PSFST

Q = TypeVar('Q')
U = TypeVar('U')
V = TypeVar('V')


def FSFST_into_PSFST(fst: FSFST[Q, U, V]) -> PSFST[Q, U, V]:
    """Convert FSFST to PSFST by normalizing frequencies to probabilities.

    Transforms a frequency-annotated SFST into a probabilistically-annotated
    SFST by dividing each frequency count by the total outgoing frequency from
    its source state. This creates a proper probability distribution where
    outgoing probabilities from each state sum to 1.0.

    Args:
        fst: A frequency-annotated SFST to convert.

    Returns:
        A probabilistically-annotated SFST.

    Raises:
        ZeroDivisionError: If any state with an outgoing transition has outgoing frequency of 0.
    """
    outgoing_freqs = outgoing_frequencies(fst)

    v, _ = fst.initial_output
    new_initial_output = v, 1.0

    new_transitions = {
        (q, u): (q_, (v, f / outgoing_freqs[q]))
        for (q, u), (q_, (v, f)) in fst.transitions.items()
    }

    new_final_outputs = {
        q: (v, f / outgoing_freqs[q])
        for q, (v, f) in fst.final_outputs.items()
    }

    return cast(PSFST[Q, U, V], PSFST(
        state_set=fst.state_set,
        input_set=fst.input_set,
        initial_state=fst.initial_state,
        transitions=new_transitions,
        initial_output=new_initial_output,
        final_outputs=new_final_outputs
    ))
