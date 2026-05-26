"""Demonstration of the RPNI algorithm learning the language of bit strings with an even number of
0s and an odd number of 1s.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from itertools import count
from automata.DFA import assert_DFA
from automata.SFST import run
from algorithms.rpni import rpni


input_set = {0, 1}


pos_dataset: list[list[int]] = [
    [1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1]
]


neg_dataset: list[list[int]] = [
    [],
    [0],
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1]
]


if __name__ == "__main__":
    dfa = rpni(
        input_set=input_set,
        pos_dataset=pos_dataset,
        neg_dataset=neg_dataset,
        choose_transition=lambda _, trs: next(iter(trs)),
        search_iter=lambda _, qs: iter(qs),
        state_supply=count(),
        verbose=True
    )

    assert_DFA(dfa)

    for d in pos_dataset:
        assert run(dfa, d, None, lambda none, _: none) != None, \
            f"learned DFA rejected positive data {d}"

    for d in neg_dataset:
        assert run(dfa, d, None, lambda none, _: none) == None, \
            f"learned DFA accepted negative data {d}"
