import numpy as np


def compress(ts, tol=0.5, max_len=np.inf):
    """
    Approximate a time series using a continuous piecewise linear function.

    Parameters
    ----------
    ts - numpy ndarray
        Time series as input of numpy array

    Returns
    -------
    pieces - numpy array
        Each row contains length, end value, increment, and error.
    """

    ts = np.asarray(ts, dtype=np.float64)
    if ts.ndim != 1 or ts.size == 0 or not np.isfinite(ts).all():
        raise ValueError("Expected a nonempty, finite one-dimensional time series.")
    if max_len < 1 or np.isnan(max_len):
        raise ValueError("max_len must be at least 1.")
    if ts.size == 1:
        return np.array([[0, ts[0], 0, 0]], dtype=np.float64)

    start = 0
    end = 1
    pieces = []
    x = np.arange(0, len(ts))
    epsilon =  np.finfo(float).eps

    while end < len(ts):
        t_st = ts[start]
        t_ed = ts[end]

        inc = t_ed - t_st
        err = t_st + (inc/(end-start))*x[0:end-start+1] - ts[start:end+1]
        err = np.inner(err, err)
        
        if end - start == 1 or (err <= tol**2 * (end - start - 1) + epsilon and end - start <= max_len):
            (last_t_ed, lastinc, lasterr) = (t_ed, inc, err) 
            end += 1
        else:
            pieces.append([end-start-1, last_t_ed, lastinc, lasterr])
            start = end - 1
            
    pieces.append([end-start-1, last_t_ed, lastinc, lasterr])

    return np.asarray(pieces, dtype=np.float64)
