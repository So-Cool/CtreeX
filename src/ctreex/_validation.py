"""Input validation helpers."""
import numpy as np

__all__ = ['check_columns', 'check_row', 'check_array_2d']


def check_columns(columns):
    if not isinstance(columns, (list, tuple)) or not columns:
        raise TypeError('columns has to be a non-empty list of (feature, '
                        'category) pairs.')
    for column in columns:
        if not isinstance(column, (list, tuple)) or len(column) != 2:
            raise TypeError('Every column has to be a (feature, category) '
                            'pair; got {!r}.'.format(column))
    return True


def check_row(row, n_columns):
    """Validates a single (one-hot encoded) data point."""
    row = np.asarray(row)
    if row.ndim == 2 and row.shape[0] == 1:
        row = row[0]
    if row.ndim != 1:
        raise ValueError('A data point has to be a 1-dimensional array.')
    if row.shape[0] != n_columns:
        raise ValueError('The data point has {} columns but {} were expected.'
                         ''.format(row.shape[0], n_columns))
    return row


def check_array_2d(array, n_columns=None):
    """Validates a 2-dimensional numerical array."""
    array = np.asarray(array)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError('The data have to be a 2-dimensional array.')
    if array.dtype.names is not None:
        raise TypeError('Structured arrays are not supported -- one-hot '
                        'encode the data into a plain numerical array.')
    if n_columns is not None and array.shape[1] != n_columns:
        raise ValueError('The data have {} columns but {} were expected.'
                         ''.format(array.shape[1], n_columns))
    return array
