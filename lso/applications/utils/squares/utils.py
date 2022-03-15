import numpy as np


def calculate_area(x: np.array) -> np.array:
    nb_of_ones = np.where(x == 1, 1, 0)
    nb_of_ones = np.sum(nb_of_ones, axis=(1, 2))
    return nb_of_ones


def calculate_circumscribed_square_area(x: np.array) -> np.array:
    row_lengths = np.sum(x, axis=2)
    col_lengths = np.sum(x, axis=1)

    if (row_lengths.shape[0] == 0) | (col_lengths.shape[0] == 0):
        return np.zeros(shape=(x.shape[0],))

    row_length_max = np.max(row_lengths, axis=1)
    col_lengths_max = np.max(col_lengths, axis=1)

    circumscribed_square_area = row_length_max * col_lengths_max

    return circumscribed_square_area
