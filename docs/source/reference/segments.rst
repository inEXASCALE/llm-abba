Segments, aggregation, and reconstruction
=========================================

.. py:currentmodule:: llmabba

These functions are available from ``llmabba`` with the Python or Cython backend. Pass double-precision NumPy arrays when handing segment buffers between functions. Some intermediate returns are NumPy arrays with the Python backend and buffer views with Cython. Use ``np.asarray()`` for array operations across both.

.. py:function:: compress(ts, tol=0.5, max_len=inf)

   Approximate a nonempty, finite, one-dimensional input with linear pieces.

   :param ts: One numeric series.
   :param tol: Segmentation tolerance used in the squared-error test.
   :param max_len: Segment length cap in observation intervals, at least one.
   :returns: A double array of shape ``(n_segments, 4)`` with columns length, end value, increment from the segment start, and sum of squared fitting errors. A one-point series produces one zero-length row.
   :raises ValueError: Invalid input shape, empty or non-finite input, or invalid length cap.

   .. testcode:: segments

      import numpy as np
      from llmabba import compress

      pieces = compress([1.0, 2.0, 3.0], tol=0.1)
      np.testing.assert_allclose(pieces, [[2.0, 3.0, 2.0, 0.0]])

.. py:function:: aggregate(data, sorting='norm', tol=0.5)

   Group rows by Euclidean distance to an ordered group starting point. Use a two-dimensional float64 array. ``sorting="norm"`` orders rows by their norm, while ``"pca"`` orders them by a principal-component score. The other values use input order in the bundled implementations.

   :param data: Array with shape ``(n_rows, n_features)``.
   :param sorting: Ordering before grouping.
   :param tol: Maximum distance from a starting point for assignment.
   :returns: ``(labels, starting_points)``. Each starting-point record contains its row index, label, member count, and the starting row coordinates.

   .. testcode:: segments

      from llmabba import aggregate

      points = np.array([[0.0, 0.0], [0.1, 0.0], [2.0, 0.0]])
      labels, groups = aggregate(points, tol=0.2)
      assert list(labels) == [0, 0, 1]
      assert [row[2] for row in groups] == [2, 1]

.. py:function:: inv_digitize(strings, centers, alphabets)

   Map each symbol to the first two columns of its corresponding prototype. Pass a nonempty sequence of symbols, a float64 prototype array, and a list of symbol names in prototype order.

   :returns: An array or view with columns length and end value.
   :raises ValueError: A symbol is absent from the alphabet, or no pieces can be assembled.

   .. testcode:: segments

      from llmabba import inv_digitize

      centers = np.array([[1.7, 2.0], [1.3, 3.0]])
      alphabet = ["rise", "high"]
      mapped = np.asarray(inv_digitize(["rise", "high"], centers, alphabet))
      np.testing.assert_allclose(mapped, centers)

.. py:function:: quantize(pieces)

   Round segment lengths onto the integer time grid and carry the rounding correction across adjacent pieces. Pass a writable, nonempty float64 buffer with a length column. The buffer is modified in place and returned as an array or view.

   .. testcode:: segments

      from llmabba import quantize

      grid_pieces = np.asarray(quantize(mapped.copy()))
      np.testing.assert_allclose(grid_pieces[:, 0], [2.0, 1.0])
      np.testing.assert_allclose(grid_pieces[:, 1], [2.0, 3.0])

.. py:function:: inv_compress(pieces, start)

   Interpolate from ``start`` to the first endpoint and connect the remaining endpoints. Pass a nonempty float64 buffer with length and end value as its first two columns. Lengths must already be valid interpolation widths. A single zero-length piece returns just the start value.

   :returns: The reconstructed series as a list.

   .. testcode:: segments

      from llmabba import inv_compress

      reconstructed = inv_compress(grid_pieces, start=0.0)
      np.testing.assert_allclose(reconstructed, [0.0, 1.0, 2.0, 3.0])

.. py:function:: inv_transform(strings, centers, alphabets, start=0)

   Combine :func:`inv_digitize`, :func:`quantize`, and :func:`inv_compress`. Use a sequence of symbols (a list for multi-character symbols), float64 prototypes, a list of symbols in prototype order, and the starting value.

   :returns: A reconstructed list. An empty symbol sequence returns ``[start]``.

   .. testcode:: segments

      from llmabba import inv_transform

      decoded = inv_transform(["rise", "high"], centers, alphabet, start=0.0)
      np.testing.assert_allclose(decoded, reconstructed)

Prototype quantization
----------------------

.. py:currentmodule:: llmabba.quant

.. py:class:: QUANT(bits=8, sign=1, zpoint=1, rd_func=None, clip_range=None, epsilon=1e-12)

   Quantize a numeric vector on an affine integer grid. An instance stores the fitted scale and offset from its most recent call. Standard widths of 8, 16, 32, and 64 bits use the corresponding signed integer dtype. Other widths are simulated in ``int64``.

   :param bits: Target bit width.
   :param sign: Select the signed-range branch with ``1``. The alternate branch requires ``zpoint=0`` and uses a symmetric input range.
   :param zpoint: Initial range selector. The instance field subsequently stores the fitted offset.
   :param rd_func: Accepted for compatibility. The implementation uses NumPy rounding.
   :param clip_range: Optional integer clipping bounds.
   :param epsilon: Accepted by the constructor. Constant inputs use a zero scale.

   .. py:method:: __call__(x)

      Fit the scale and return integer codes for the finite nonempty numeric array ``x``.

   .. py:method:: dequant(x_q)

      Decode integer codes using the scale and offset from the previous call.

   .. py:method:: compute_scaling(alpha, beta, alpha_q, beta_q)

      Map input bounds ``[alpha, beta]`` to integer-grid bounds ``[alpha_q, beta_q]`` and return ``(scale, offset)``. The integer bounds must differ.

   .. py:method:: quantization(x, s, z)

      Quantize with an explicit nonzero scale ``s`` and offset ``z``. Returns clipped, rounded integer codes.

   .. py:method:: dequantization(x_q, s, z)

      Decode with an explicit scale and offset, returning floating-point values.

   The example exercises the fitted and explicit-scale forms.

   .. testcode::

      import numpy as np
      from llmabba.quant import QUANT

      values = np.array([1.0, 2.0, 3.0])
      quantizer = QUANT(bits=8)
      codes = quantizer(values)
      np.testing.assert_allclose(quantizer.dequant(codes), values, atol=0.01)

      scale, offset = quantizer.compute_scaling(1.0, 3.0, -128, 127)
      explicit_codes = quantizer.quantization(values, scale, offset)
      np.testing.assert_array_equal(explicit_codes, codes)
      np.testing.assert_allclose(
          quantizer.dequantization(explicit_codes, scale, offset), values, atol=0.01,
      )
