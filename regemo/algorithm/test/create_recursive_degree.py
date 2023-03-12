import numpy as np
import copy
from math import comb


def create_list_degrees(max_degree=3,
                        num_vars=3,
                        cur_list=None,
                        full_list=None):
    if full_list is None:
        full_list = []
    if cur_list is None:
        cur_list = []

    if len(cur_list) == num_vars:
        # base check if the cur list involves all the variables
        full_list.append(cur_list)
        return

    cur_sum = 0 if len(cur_list) == 0 else np.sum(cur_list)
    for i in range(max_degree - cur_sum + 1):
        cur_list_copy = copy.deepcopy(cur_list)
        cur_list_copy.append(i)
        create_list_degrees(max_degree=max_degree,
                            num_vars=num_vars,
                            cur_list=cur_list_copy,
                            full_list=full_list)

    # return the full list
    return full_list


def main():
    list_degrees = create_list_degrees(max_degree=2, num_vars=2)
    print("here")


if __name__ == "__main__":
    main()
