import numpy as np
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints


def get_knee(f, epsilon=0.125):
    # KLUGE: Pymoo changed from original. Line 23 in mcdm.high_tradeoff.py
    # neighbors = neighbors_finder.find(i) replaced by the next line
    # neighbors = [j for j in neighbors_finder.find(i) if j != i]

    # dm = HighTradeoffPoints(epsilon=epsilon)
    dm = HighTradeoffPoints(zero_to_one=True, ideal=np.array([0, 0]), nadir=np.array([3, 0.012]))
    knee_indx = dm.do(f)

    return knee_indx