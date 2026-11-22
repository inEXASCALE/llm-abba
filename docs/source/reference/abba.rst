ABBA quick interface
====================

.. py:currentmodule:: llmabba

.. py:class:: ABBA(**args)

   Encode and reconstruct numerical signals with a codebook learned from their segments. All keyword arguments are forwarded to :class:`XABBA`. Use ``tol`` to control segmentation and ``alpha`` to control aggregation. Use :class:`XABBA` directly for explicit control over the fit and transform stages.

   .. testcode:: abba

      from llmabba import ABBA

      encoder = ABBA(tol=0.1, alpha=0.1, verbose=0)

   .. py:method:: encode(ts)

      Fit a new codebook and encode the supplied signal or batch. For a batch, pass equal-length rows of numeric values. For multivariate data, use an array of shape ``(samples, time, features)``. A one-dimensional input is treated as a single series and can be partitioned for compression.

      :param ts: Numeric series or rectangular batch of series.
      :returns: One list of symbols per input row or one-dimensional partition. The codebook and row start values are stored on the encoder.

      .. testcode:: abba

         signals = [[1.0, 1.5, 2.0], [2.0, 3.0, 4.0]]
         symbols = encoder.encode(signals)
         assert len(symbols) == 2

   .. py:method:: decode(strings)

      Reconstruct the sequences returned by the most recent ``encode()`` using the fitted codebook and stored start values.

      :param strings: Sequences of symbols in the fitted alphabet.
      :returns: A list of reconstructed rows for a batch, or a flat NumPy array for a one-dimensional series. A multivariate batch still has flat reconstructed rows. See :meth:`XABBA.recast_shape` to recover the dimensions.
      :raises ValueError: The count of sequences differs from the count of stored starting values.

      .. testcode:: abba

         reconstructed = encoder.decode(symbols)
         assert len(reconstructed) == len(signals)

   The inherited :meth:`XABBA.transform` encodes new data with the current codebook. Pass its returned start values explicitly to :meth:`XABBA.inverse_transform` when reconstructing the new sequences.
