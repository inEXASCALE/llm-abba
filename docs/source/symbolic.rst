Symbolic approximation
======================

Choose an encoder
-----------------

* :class:`llmabba.ABBA` provides ``encode()`` and ``decode()`` for a short CPU workflow.
* :class:`llmabba.XABBA` exposes fit, transform, and inverse transform separately. Use it to keep a codebook fixed across splits.
* :doc:`reference/fast` documents the single-series variants and the prototype columns each quantizes.

The fixed-point adaptive piecewise linear approximation used here groups segments using their lengths and end values. The decoder joins segment endpoints beginning at the supplied initial value. A piece also records its increment and fitting error for low-level analysis.

Fit once and reuse the codebook
-------------------------------

.. testcode::

   import numpy as np
   from llmabba import XABBA

   training = [[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]]
   unseen = [[0.0, 1.1, 2.0]]

   encoder = XABBA(tol=0.1, alpha=0.1, verbose=0)
   encoder.fit(training, n_jobs=1)
   symbols, starts = encoder.transform(unseen, n_jobs=1)
   reconstructed = encoder.inverse_transform(symbols, start_set=starts)
   np.testing.assert_allclose(reconstructed, [[0.0, 1.0, 2.0]], atol=1e-3)

``fit_transform()`` returns the assignments produced while fitting. ``transform()`` assigns each new segment to a fitted prototype. If both development and test symbols should come from the nearest-prototype rule, call ``fit()`` once and ``transform()`` on every split. Keep the start values returned with the transformed symbols.

Handle multivariate signals
---------------------------

XABBA flattens the trailing axes of each sample in row-major order. Fit on an array of shape ``(samples, time, features)`` and supply the same time and feature dimensions when transforming new samples. Inverse transformation returns flat rows. If a fixed tensor shape is needed, ``recast_shape()`` pads or truncates the reconstructed rows and reshapes them.

Control approximation and vocabulary size
------------------------------------------

* ``tol`` controls when a segment ends. A smaller tolerance usually yields more segments.
* With ``init="agg"``, ``alpha`` is the grouping tolerance in scaled segment space. A larger tolerance usually yields fewer symbols.
* With ``init="kmeans"`` or ``init="f-kmeans"``, ``k`` requests a number of clusters.
* ``scl`` weights segment length in the grouping distance. ``bits_for_len`` and ``bits_for_inc`` control the precision of stored prototype lengths and end values.

When comparing settings, record the number of segments, vocabulary size, and reconstruction error. Compare downstream settings on validation data. The `fABBA project <https://github.com/nla-group/fABBA>`_ and :doc:`references` describe the underlying methods.
