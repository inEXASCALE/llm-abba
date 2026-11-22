Single-series encoders
======================

.. py:currentmodule:: llmabba

The three variants use one finite one-dimensional series at a time. Each call to ``transform()`` fits a fresh codebook. For a single codebook shared across a batch, use :class:`XABBA`.

.. autoclass:: llmabba.fastXABBA

   Stores quantized lengths and end values. The constructor arguments are shared by the three variants.

   .. list-table::
      :header-rows: 1

      * - Arguments
        - Meaning
      * - ``tol``, ``max_len``
        - Segmentation tolerance and maximum segment length.
      * - ``init``, ``alpha``, ``sorting``
        - Choose ``"agg"`` with an aggregation tolerance and ordering, or ``"kmeans"`` for sampled k-means.
      * - ``k``, ``r``, ``max_iter``, ``random_state``
        - Cluster count, sampling fraction, iteration limit, and seed for k-means. Choose ``k`` at most the number of compressed pieces.
      * - ``scl``, ``bits_for_len``, ``bits_for_inc``
        - Relative weight of lengths and bit widths for the prototype columns described on this page.
      * - ``n_jobs``, ``alphabet_set``, ``verbose``
        - Worker count for k-means, alphabet selector or list as in :meth:`XABBA.fit`, and whether to print the learned vocabulary size.

   .. py:method:: transform(series)

      Fit on a one-dimensional numeric series and return its symbols as a list. This replaces the previous codebook.

   .. py:method:: predict(series)

      Use the fitted k-means codebook to assign symbols to a new series and return a one-dimensional array of symbols. Call ``transform()`` with ``init="kmeans"`` first. The default aggregation mode raises ``ValueError`` for prediction.

   .. py:method:: inverse_transform(strings, start)

      Decode fitted alphabet symbols beginning at ``start``. Returns a list of approximated values. Use the starting value associated with the encoded input.

   See :ref:`single-series-example` for calls to all three methods.

.. autoclass:: llmabba.fastXABBA_len

   Quantizes the length column only. The end-value column is kept as floating-point values.

   .. py:method:: transform(series)

      Fit and return symbols from one finite series, quantizing the stored lengths.

   .. py:method:: predict(series)

      Return symbols for a new series after a k-means fit. The restrictions of :meth:`fastXABBA.predict` also apply here.

   .. py:method:: inverse_transform(strings, start)

      Reconstruct a list of values using the decoded lengths and floating-point end values.

   See :ref:`single-series-example`.

.. autoclass:: llmabba.fastXABBA_inc

   Quantizes the end-value column only. The length column is kept as floating-point values. The bit-width option for end values is named ``bits_for_inc``.

   .. py:method:: transform(series)

      Fit and return symbols from one finite series, quantizing the stored end values.

   .. py:method:: predict(series)

      Return symbols for a new series after a k-means fit. The restrictions of :meth:`fastXABBA.predict` also apply here.

   .. py:method:: inverse_transform(strings, start)

      Reconstruct a list of values using the stored lengths and decoded end values.

   See :ref:`single-series-example`.

.. _single-series-example:

Fit, predict, and reconstruct
------------------------------

This example runs the three methods for each encoder. A simple piecewise linear input makes the length and starting value easy to check.

.. testcode::

   import numpy as np
   from llmabba import fastXABBA, fastXABBA_len, fastXABBA_inc

   values = [0.0, 1.0, 2.0]
   for encoder_type in (fastXABBA, fastXABBA_len, fastXABBA_inc):
       encoder = encoder_type(init="kmeans", k=1, verbose=False)
       fitted_symbols = encoder.transform(values)
       symbols = encoder.predict(values)
       reconstructed = encoder.inverse_transform(symbols, start=values[0])
       assert list(symbols) == fitted_symbols
       np.testing.assert_allclose(reconstructed, values, atol=1e-3)
