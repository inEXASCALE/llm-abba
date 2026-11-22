Input data and splits
=====================

Array shapes
------------

Use floating-point signal samples in chronological order. The batch axis identifies independent examples. Keep the feature order the same during development and inference.

.. list-table::
   :header-rows: 1

   * - Task
     - Input to ``process()``
     - Input to ``inference()``
     - Inference output
   * - Classification
     - Dictionary with ``X_data`` and one class label per sample in ``Y_data``. Use ``(samples, time)`` or ``(samples, time, features)`` for arrays.
     - One sample or a batch with the shape used during fitting.
     - A label string for one sample or a list of label strings for a batch.
   * - Regression
     - The same feature shapes, with one numeric target per sample. Nested pandas frames from the ``.ts`` loader are also supported.
     - One sample or a batch. For multivariate arrays use ``(time, features)`` or ``(samples, time, features)``.
     - A generated string for one sample or a list of strings in input order.
   * - Forecasting
     - A chronological array of shape ``(time, features)``.
     - One history of shape ``(seq_len_pre, features)``.
     - A numeric array of shape ``(seq_len_post, features)`` in the original units.

For a univariate supervised task, a row represents one whole recording. For a multivariate supervised task, each time step contains all feature values for that step. For a univariate forecast, keep the feature axis with ``values.reshape(-1, 1)``.

.. testcode::

   import numpy as np

   signals = np.array([[0.0, 0.3, 0.4], [1.0, 0.5, 0.0]])
   labels = np.array(["rising", "falling"])
   classification = {"X_data": signals, "Y_data": labels}
   single_sample = classification["X_data"][:1]

   temperatures = np.linspace(18.0, 25.0, 96).reshape(-1, 1)
   history = temperatures[-24:]
   assert single_sample.shape == (1, 3)
   assert history.shape == (24, 1)

For a nested frame, each cell contains a channel as a pandas Series. :func:`llmabba.utils.regressor_tools.process_data` aligns channels within a recording and returns an array in ``(samples, time, features)`` order. Select a target length from the training data and reuse it for later splits.

Missing values and scaling
--------------------------

Keep labels, timestamps, and channel names out of the numeric array. The low-level :func:`llmabba.compress` requires finite samples. XABBA can fill missing values using its ``fillna`` setting. The nested-frame regression helper interpolates missing channel values where possible. Check for channels with no observations before training.

With ``scalar="z-score"``, supervised two-dimensional inputs are standardized per time position across training recordings. For three-dimensional inputs, statistics are per channel across training recordings and time steps. Forecasting computes per-channel statistics from the training time range. ``scalar="min-max"`` uses the same axes with min-max scaling.

Development and test data
-------------------------

* **Classification**. Keep labels outside the feature matrix. Make the outer holdout at the sample level and stratify by class where appropriate. Use a group split if multiple recordings from the same subject must stay together.
* **Regression**. Use the existing train and test files when the dataset supplies them. Fit any external preprocessing from training data and reuse it on the test set.
* **Forecasting**. Reserve a later time range before creating windows. An input history ends immediately before its target horizon begins.

For supervised inputs, ``process()`` internally makes a random 80/20 training and validation split. For forecasting, it makes a chronological split with separate time ranges. It fits the scaler and codebook using the training portion and reuses them for validation. Reserve independent test data before calling it. See :meth:`llmabba.LLMABBA.process` for arguments and artifact paths.
