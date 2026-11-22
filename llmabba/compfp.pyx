cimport cython
import numpy as np
cimport numpy as np
np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.binding(True)

cpdef compress(ts, double tol=0.5, double max_len=np.inf):
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

    values = np.asarray(ts, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty, finite one-dimensional time series.")
    if max_len < 1 or np.isnan(max_len):
        raise ValueError("max_len must be at least 1.")
    if values.size == 1:
        return np.array([[0, values[0], 0, 0]], dtype=np.float64)

    cdef const double[:] series = values
    cdef int start = 0
    cdef int end = 1
    cdef int i, len_t = len(series)
    cdef list pieces = []
    cdef double epsilon =  np.finfo(float).eps
    
    cdef double t_st, t_ed, last_t_ed, slope, residual
    cdef double inc, lastinc, err, lasterr
    
    while end < len_t:
        t_st = series[start]
        t_ed = series[end]
        inc = t_ed - t_st
        err = 0.0
        
        slope = inc / (end - start)
        for i in range(end - start + 1):
            residual = t_st + slope * i - series[start + i]
            err += residual * residual
        
        if end - start == 1 or (err <= tol**2 * (end - start - 1) + epsilon and end - start <= max_len):
            (last_t_ed, lastinc, lasterr) = (t_ed, inc, err)
            end = end + 1
        else:
            pieces.append([end-start-1, last_t_ed, lastinc, lasterr])
            start = end - 1

    pieces.append([end-start-1, last_t_ed, lastinc, lasterr])

    return np.asarray(pieces, dtype=np.float64)
