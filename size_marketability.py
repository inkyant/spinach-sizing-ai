
import numpy as np

SMALL_SIZE_CUTOFF = 5
LARGE_SIZE_CUTOFF = 20

POOR_SIZE_VALUE = 0.25
GOOD_SIZE_VALUE = 0.5


def get_markertability_by_size(spinach_sizes):
    return np.vectorize(value_func)(spinach_sizes)

def value_func(spinach_leaf_size):
    if SMALL_SIZE_CUTOFF <= spinach_leaf_size <= LARGE_SIZE_CUTOFF:
        return spinach_leaf_size*GOOD_SIZE_VALUE
    else:
        return spinach_leaf_size*POOR_SIZE_VALUE


if __name__ == "__main__":
    # just test with normal distribution, though this gives negative values sometimes
    arr = np.random.normal(15, 10, 5)
    print(arr)
    print()
    print(get_markertability_by_size(arr))