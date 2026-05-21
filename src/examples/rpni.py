"""Demonstration of the RPNI algorithm learning the language of bit strings with an even number of
0s and an odd number of 1s.
"""
from itertools import count
from algorithms.rpni import rpni


input_set = {0, 1}


pos_dataset: list[list[int]] = [
    [1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1, 0, 1]
]


neg_dataset: list[list[int]] = [
    [],
    [0],
    [0, 0],
    [0, 1]
]


if __name__ == "__main__":
    rpni(
        input_set=input_set,
        pos_dataset=pos_dataset,
        neg_dataset=neg_dataset,
        choose_transition=lambda _, trs : next(iter(trs)),
        search_iter=lambda _, qs : iter(qs),
        state_supply=count(),
        verbose=True
    )
