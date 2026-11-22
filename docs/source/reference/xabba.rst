XABBA and fitted codebooks
==========================

.. py:currentmodule:: llmabba

.. py:class:: XABBA(tol=0.2, init='agg', k=2, r=0.5, alpha=0.1, sorting='norm', scl=1, max_iter=2, bits_for_len=8, bits_for_inc=16, partition_rate=None, partition=None, max_len=inf, verbose=1, random_state=42, fillna='ffill')

   Fit and reuse a symbolic codebook. Each prototype stores a segment length and end value. Public attributes include ``parameters`` (encoded prototypes and symbols), ``start_set`` (training row starts), ``string_`` (training sequences), and ``recap_shape`` (the original shape for inputs with more than two dimensions).

   .. list-table::
      :header-rows: 1

      * - Arguments
        - Meaning
      * - ``tol``, ``max_len``
        - Positive segmentation tolerance and maximum segment length in intervals. ``max_len`` must be at least one.
      * - ``init``, ``alpha``, ``sorting``
        - Choose ``"agg"``, ``"kmeans"``, or ``"f-kmeans"``. Aggregation uses positive distance tolerance ``alpha`` and an ordering such as ``"norm"`` or ``"pca"``.
      * - ``k``, ``r``, ``max_iter``, ``random_state``
        - Requested cluster count for k-means, sample fraction for sampled k-means, its iteration limit and sampling seed. The fitted number of distinct symbols can differ from the request.
      * - ``scl``, ``bits_for_len``, ``bits_for_inc``
        - Nonnegative relative weight of segment lengths and the quantization widths for the stored lengths and end values.
      * - ``partition``, ``partition_rate``
        - For a one-dimensional input, set a positive partition count or a rate in ``(0, 1]``. An explicit count takes precedence. The implementation limits the number of partitions for short signals.
      * - ``fillna``, ``verbose``
        - Choose ``"ffill"``, ``"bfill"``, ``"mean"``, ``"median"``, or ``"zero"`` for missing values. Set verbosity to zero to suppress fitting messages.

   The implementation accepts ``sorting`` values ``"norm"``, ``"pca"``, ``"lexi"``, ``"1-norm"``, and ``"2-norm"``. The bundled aggregation kernels provide dedicated sorting for ``"norm"`` and ``"pca"``. The remaining values fall through to input order.

   .. testcode:: xabba

      import numpy as np
      from llmabba import XABBA

      training = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
      encoder = XABBA(tol=0.1, alpha=0.1, scl=2, verbose=0)
      assert encoder.tol == 0.1 and encoder.scl == 2

   .. py:method:: fit(series, n_jobs=-1, alphabet_set=0)

      Learn a codebook from one signal or a batch. A rectangular two-dimensional input supplies a series per row. Higher-dimensional arrays flatten the trailing axes per sample.

      :param series: Nonempty, numeric input.
      :param n_jobs: Worker count. A negative value selects available CPUs, capped by workload. Use one for serial execution.
      :param alphabet_set: Zero for interleaved upper- and lowercase letters, one for uppercase first, minus one for lowercase first, or a list of symbols.
      :returns: The fitted encoder.

      .. testcode:: xabba

         assert encoder.fit(training, n_jobs=1, alphabet_set=["rise", "steep"]) is encoder
         assert len(encoder.parameters.alphabets) == 2

   .. py:method:: fit_transform(series, n_jobs=-1, alphabet_set=0, return_start_set=False)

      Fit and return the assignments made during fitting. The first three arguments match :meth:`fit`.

      :param return_start_set: Include one starting value per symbolic sequence.
      :returns: A list of symbolic sequences, or ``(sequences, starts)`` when requested.

      .. testcode:: xabba

         sequences, starts = encoder.fit_transform(
             training, n_jobs=1, return_start_set=True,
         )
         assert len(sequences) == len(starts) == 2

   .. py:method:: transform(series, n_jobs=-1, llm_split='Pre')

      Assign pieces of new series to the nearest fitted prototypes. Fit or restore an encoder first. For fitted multivariate data, keep the trailing dimensions unchanged. ``llm_split="Post"`` is used for forecast targets with a different horizon.

      :param series: New numeric input.
      :param n_jobs: Worker count as in :meth:`fit`.
      :param llm_split: Either ``"Pre"`` or ``"Post"``.
      :returns: ``(sequences, starts)``, in input row or partition order.
      :raises ValueError: Empty input, invalid split, or incompatible multivariate dimensions.

      .. testcode:: xabba

         sequences, starts = encoder.transform([[0.0, 1.1, 2.0]], n_jobs=1)
         assert len(sequences) == len(starts) == 1

   .. py:method:: inverse_transform(string_sequences, start_set=None, n_jobs=1)

      Decode with the fitted codebook. Symbols must belong to its alphabet. Supply the new starts from :meth:`transform` when decoding newly transformed inputs.

      :param string_sequences: One sequence of symbols per row or partition.
      :param start_set: One initial value per sequence. The default uses the training start values.
      :param n_jobs: Number of reconstruction workers. The default is serial.
      :returns: Reconstructed rows as a list, or a concatenated one-dimensional array when the most recent input was one-dimensional.
      :raises ValueError: A start value is missing for a sequence or a symbol cannot be resolved.

      .. testcode:: xabba

         rows = encoder.inverse_transform(sequences, start_set=starts)
         np.testing.assert_allclose(rows, [[0.0, 1.0, 2.0]], atol=1e-3)

   .. py:method:: recast_shape(reconstruct_list, recap_shape=None)

      Reshape flat reconstructed rows. The default uses the trailing dimensions captured at fit time. Pass a target shape when ``recap_shape`` was not captured. The returned batch size is the number of supplied rows.

      :param reconstruct_list: Sequence of flat rows.
      :param recap_shape: Target shape with batch axis first.
      :returns: A NumPy array with the requested trailing dimensions. Short rows are padded with the row mean, or zero for an empty row. Long rows are truncated. Adjustment emits ``ShapeWarning``.

      .. testcode:: xabba

         batch = encoder.recast_shape(rows, recap_shape=(1, 3, 1))
         assert batch.shape == (1, 3, 1)

   .. rubric:: Lower-level encoder methods

   These hooks expose compression and assignment for custom pipelines. :meth:`fit` combines compression and digitization automatically.

   .. py:method:: parallel_compress(series, n_jobs=-1)

      Prepare the input as in :meth:`fit`, store original row starts, and return a list of segment arrays. Each array has the columns described by :func:`compress`.

      .. testcode:: xabba

         pieces = encoder.parallel_compress(training, n_jobs=1)
         assert len(pieces) == 2 and pieces[0].shape[1] == 4

   .. py:method:: digitize(series, pieces, alphabet_set=0, n_jobs=-1)

      Group the arrays from :meth:`parallel_compress` and store the quantized prototypes and alphabet. Pass the original numeric input as ``series``, the per-row arrays as ``pieces``, and the worker and alphabet options from :meth:`fit`. Return one sequence per piece array.

      .. testcode:: xabba

         sequences = encoder.digitize(training, pieces, alphabet_set=0, n_jobs=1)
         assert len(sequences) == len(pieces)

   .. py:method:: transform_single_series(series)

      Compress one finite one-dimensional series and return its list of fitted symbols. The caller handles input preparation and start values.

      .. testcode:: xabba

         symbols = encoder.transform_single_series(training[0])
         assert len(symbols) == len(pieces[0])

   .. py:method:: piece_to_symbol(piece)

      Match a two-element ``(length, end_value)`` piece to the nearest prototype using the fitted scale and return its symbol.

      .. testcode:: xabba

         symbol = encoder.piece_to_symbol(pieces[0][0, :2])
         assert symbol in encoder.parameters.alphabets

   .. py:method:: string_separation(symbols, num_pieces)

      Split a flat list of symbols into lists by the lengths in ``num_pieces``. The counts should sum to the number of symbols.

      .. testcode:: xabba

         assert encoder.string_separation(["a", "b", "c"], [2, 1]) == [["a", "b"], ["c"]]

   .. py:method:: n_jobs_init(n_jobs=-1, _max=inf)

      Resolve available workers for negative or ``None`` values and cap the result at ``_max``. Non-integers and zero are rejected.

      .. testcode:: xabba

         assert encoder.n_jobs_init(4, _max=2) == 2

   .. rubric:: Configuration attributes

   The attributes ``tol``, ``alpha``, ``sorting``, ``scl``, ``max_len``, ``partition``, ``partition_rate``, and ``k`` expose the corresponding constructor settings. Set them before fitting to keep codebook construction and assignment consistent. The read/write ``eta`` attribute is assigned by digitization and can be inspected after fitting.

   .. testcode:: xabba

      configuration = XABBA(verbose=0)
      configuration.tol = 0.1
      configuration.alpha = 0.2
      configuration.sorting = "norm"
      configuration.scl = 2
      configuration.max_len = 8
      configuration.partition = 2
      configuration.partition_rate = 0.5
      configuration.k = 2
      configuration.fit(training, n_jobs=1)
      assert configuration.eta is not None
