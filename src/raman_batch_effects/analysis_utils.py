import numpy as np


def find_elbow_by_max_distance(y):
    """
    Find the index of the 'elbow' (knee) in a monotonically increasing curve
    using the maximum distance to the line connecting the first and last points.

    Parameters
    ----------
    y : array-like of shape (n,), sorted in ascending order.

    Returns
    -------
    elbow_index : int
        Index of the elbow point in y (after sorting).
    """
    eps = 1e-12

    y = np.asarray(y, dtype=float)

    num_points = len(y)
    if num_points < 3:
        raise ValueError("Need at least 3 points to find an elbow")

    # Normalize x and y to [0, 1] for scale invariance.
    x = np.linspace(0, 1, num_points)
    y_norm = (y - y[0]) / (y[-1] - y[0] + eps)

    # The line from the first to the last point.
    x0, y0 = x[0], y_norm[0]
    x1, y1 = x[-1], y_norm[-1]

    # Compute perpendicular distances from each point to the line.
    numerator = np.abs((y1 - y0) * x - (x1 - x0) * y_norm + x1 * y0 - y1 * x0)
    denominator = np.hypot(y1 - y0, x1 - x0)
    distances = numerator / (denominator + eps)

    elbow_index = int(np.argmax(distances))

    return elbow_index
