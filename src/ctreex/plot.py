"""
Plots of leaf-to-leaf distances (requires the ``plot`` extra: matplotlib).

Reimplementation of the original ``metric_heat_map``, ``leaf_metric_barplot``
and ``leaf_metric_histogram`` helpers.
"""
import numpy as np

from ctreex.counterfactuals import same_leaves

__all__ = ['metric_heat_map', 'leaf_metric_barplot', 'leaf_metric_histogram']


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError('Plotting requires matplotlib: '
                          'pip install "ctreex[plot]".') from e
    return plt


def _order_by_class(leaf_classes):
    # leaves sorted by class (descending), ties kept in their original order
    return sorted(range(len(leaf_classes)), key=lambda i: leaf_classes[i],
                  reverse=True)


def metric_heat_map(matrix, leaf_ids, leaf_classes, class_case=False,
                    annot=False, xtick=False, ytick=False, ax=None):
    """
    Heat map of a leaf-to-leaf distance matrix with leaves grouped by class
    (sorted by their values for regression trees).

    Parameters
    ----------
    class_case : bool
        Print the leaves of every class.
    annot : bool
        Write the distances in the cells.
    xtick, ytick : bool
        Label the axes with leaf ids.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _plt()
    leaf_ids, leaf_classes = np.asarray(leaf_ids), np.asarray(leaf_classes)
    order = _order_by_class(leaf_classes.tolist())
    sorted_matrix = np.asarray(matrix)[np.ix_(order, order)]
    if class_case:
        for c in sorted(set(leaf_classes.tolist()), reverse=True):
            print('{}:'.format(c))
            print('    {}'.format(leaf_ids[leaf_classes == c].tolist()))
    if ax is None:
        _, ax = plt.subplots()
    image = ax.imshow(sorted_matrix, cmap='rocket' if 'rocket' in
                      plt.colormaps() else 'magma', aspect='auto')
    ax.figure.colorbar(image, ax=ax)
    labels = leaf_ids[order].tolist()
    ax.set_xticks(range(len(labels)) if xtick else [])
    ax.set_yticks(range(len(labels)) if ytick else [])
    if xtick:
        ax.set_xticklabels(labels, rotation=90)
    if ytick:
        ax.set_yticklabels(labels)
    if annot:
        for i in range(sorted_matrix.shape[0]):
            for j in range(sorted_matrix.shape[1]):
                ax.text(j, i, '{:g}'.format(sorted_matrix[i, j]),
                        ha='center', va='center', fontsize=8, color='white')
    return ax


def _groups(leaf_id, leaf_ids, leaf_classes, tolerance):
    """``(label, mask)`` pairs: one per class, or -- with a tolerance -- the
    leaves the same as ``leaf_id`` and the others."""
    if tolerance is None:
        return [(str(c), leaf_classes == c)
                for c in sorted(set(leaf_classes.tolist()))]
    reference = leaf_classes[int(np.flatnonzero(leaf_ids == leaf_id)[0])]
    same = same_leaves(reference, leaf_classes, tolerance=tolerance)
    return [('within {:g} of {:g}'.format(tolerance, float(reference)), same),
            ('other', ~same)]


def leaf_metric_barplot(leaf_id, matrix, leaf_ids, leaf_classes,
                        tolerance=None, ax=None):
    """
    Stacked bar plot: number of leaves of every class at every distance
    from ``leaf_id``.

    For regression trees (``leaf_classes`` holds values) give a
    ``tolerance``: the bars then count the leaves whose values differ from the
    value of ``leaf_id`` by at most ``tolerance`` and the other leaves.
    """
    plt = _plt()
    leaf_ids, leaf_classes = np.asarray(leaf_ids), np.asarray(leaf_classes)
    column = np.asarray(matrix)[:, int(np.flatnonzero(leaf_ids == leaf_id)[0])]
    values = sorted(set(column.tolist()))
    if ax is None:
        _, ax = plt.subplots()
    bottom = np.zeros(len(values))
    for label, members in _groups(leaf_id, leaf_ids, leaf_classes, tolerance):
        counts = np.array([np.sum((column == v) & members) for v in values],
                          dtype=float)
        ax.bar(range(len(values)), counts, bottom=bottom, label=label)
        bottom += counts
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(['{:g}'.format(v) for v in values], rotation=90)
    ax.legend()
    return ax


def leaf_metric_histogram(leaf_id, matrix, leaf_ids, leaf_classes,
                          tolerance=None, ax=None):
    """
    Overlaid histograms (one per class) of the (whole-number) distances
    from ``leaf_id`` to all the leaves; one bin per distance.

    For regression trees give a ``tolerance`` (see
    :func:`leaf_metric_barplot`).
    """
    plt = _plt()
    leaf_ids, leaf_classes = np.asarray(leaf_ids), np.asarray(leaf_classes)
    column = np.asarray(matrix)[:, int(np.flatnonzero(leaf_ids == leaf_id)[0])]
    bins = np.arange(np.floor(column.min()), np.ceil(column.max()) + 2)
    if ax is None:
        _, ax = plt.subplots()
    for label, members in _groups(leaf_id, leaf_ids, leaf_classes, tolerance):
        ax.hist(column[members], bins=bins, alpha=0.4, align='left',
                rwidth=0.7, label=label)
    ax.legend()
    ax.set_ylabel('Count')
    ax.set_xlabel('Distance')
    return ax
