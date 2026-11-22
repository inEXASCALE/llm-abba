Developer and legacy utilities
==============================

These helpers support experiment scripts and data preparation. Functions with optional or external dependencies list the imports they require. For the core user workflow, start with :doc:`../symbolic` or :doc:`../workflows`.

Resampling and functional data
------------------------------

.. py:currentmodule:: llmabba.utils.data_processor

.. py:function:: uniform_scaling(data, max_len)

   Resample a nonempty indexable series to ``max_len`` entries using samples at uniformly spaced indices rounded down. Returns a list. Use a positive output length.

   .. testcode::

      from llmabba.utils.data_processor import uniform_scaling

      assert uniform_scaling([10, 20], max_len=4) == [10, 10, 20, 20]

.. py:function:: to_fd(series, sample_points=None)

   Convert a rectangular signal batch of shape ``(samples, time)`` to ``skfda.FDataGrid``. The optional grid has one coordinate per time step. The default is integer positions. This helper requires the optional ``scikit-fda`` dependency.

   .. code-block:: python

      import numpy as np
      from llmabba.utils.data_processor import to_fd

      signals = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
      functional = to_fd(signals, sample_points=[0.0, 0.5, 1.0])

Text and collection helpers
---------------------------

.. py:currentmodule:: llmabba.utils.fundamentals

.. py:function:: listToString(s)

   Append one space after each input string, including the last. Returns the joined string.

   .. testcode::

      from llmabba.utils.fundamentals import listToString

      assert listToString(["A", "b"]) == "A b "

.. py:function:: listToString_blank(s)

   Concatenate string elements without separators.

   .. testcode::

      from llmabba.utils.fundamentals import listToString_blank

      assert listToString_blank(["A", "b"]) == "Ab"

.. py:function:: stringToList(s)

   Read one-character symbols from alternating character positions. Pass text in the format produced by ``listToString()`` for single-character symbols. For variable-length symbols, see :func:`llmabba.llm_abba.StringToList`.

   .. testcode::

      from llmabba.utils.fundamentals import stringToList

      assert stringToList("A b ") == ["A", "b"]

.. py:function:: mean(arr)

   Return the arithmetic mean of a nonempty numeric sequence.

   .. testcode::

      from llmabba.utils.fundamentals import mean

      assert mean([2.0, 4.0]) == 3.0

.. py:function:: cross_correlation(x, y)

   Return zero-lag correlation for equal-length aligned numeric inputs with nonzero variance. The matching LLMABBA helper is documented in :doc:`artifacts`.

   .. testcode::

      import math
      from llmabba.utils.fundamentals import cross_correlation

      assert math.isclose(cross_correlation([1.0, 2.0], [3.0, 1.0]), -1.0)

.. py:function:: find_keys_by_value(dictionary, value)

   Return keys whose values equal ``value``, in dictionary iteration order.

   .. testcode::

      from llmabba.utils.fundamentals import find_keys_by_value

      assert find_keys_by_value({"a": 2, "b": 1, "c": 2}, 2) == ["a", "c"]

.. py:function:: find_values_by_key(dictionary, keys)

   Return values in the requested key order. A missing key raises ``KeyError``.

   .. testcode::

      from llmabba.utils.fundamentals import find_values_by_key

      assert find_values_by_key({"a": 2, "b": 1}, ["b", "a"]) == [1, 2]

fNIRS experiment preprocessing
------------------------------

.. py:function:: fnirs_process(current_dir, input_folder)

   Read CSV file names in ``input_folder`` relative to the directory string ``current_dir``. Skip the first row and read the last column as labels. From each complete 426-row block, keep two 210-row windows after trimming the block boundaries. The return value is ``(signals, labels)`` with signal axes ``(samples, features, time)``. Label each window from its first row.

.. py:function:: fnirs_process_slide_window(current_dir, input_folder)

   Read CSV files with two trailing metadata columns, with the label in the last column. Read the middle label from each complete 150-row window, keep source labels 0 and 3, and map them to 0 and 1. The return value is ``(signals, labels)``. With multiple windows per class and multiple features the signal axes are ``(samples, time, features)``. The legacy squeeze operation can remove singleton axes, so check the shape before using smaller inputs.

Both examples use a temporary synthetic CSV with numeric columns.

.. testcode::

   from pathlib import Path
   from tempfile import TemporaryDirectory

   import numpy as np
   import pandas as pd
   from llmabba.utils.fundamentals import fnirs_process, fnirs_process_slide_window

   frame = pd.DataFrame({
       "feature_1": np.arange(600),
       "feature_2": np.arange(600) * 2,
       "metadata": np.zeros(600),
       "label": np.repeat([0, 3, 0, 3], 150),
   })
   with TemporaryDirectory() as directory:
       frame.to_csv(Path(directory) / "toy.csv", index=False)
       signals, labels = fnirs_process(directory, ["toy.csv"])
       windows, classes = fnirs_process_slide_window(directory, ["toy.csv"])
   assert signals.shape == (2, 3, 210)
   assert labels.tolist() == [0, 3]
   assert windows.shape == (4, 150, 2)
   assert classes.tolist() == [0, 0, 1, 1]

Processes, paths, and timings
-----------------------------

.. py:currentmodule:: llmabba.utils.tools

.. py:function:: initialise_multithread(num_cores=-1)

   Return a multiprocessing pool. When ``num_cores`` is minus one or at least the reported CPU count, the helper requests one fewer worker than the reported count. Use an explicit positive worker count smaller than the available CPU count. Close the pool after use.

   .. testcode::

      from llmabba.utils.tools import initialise_multithread

      with initialise_multithread(1) as pool:
          assert pool.map(abs, [-1, 2]) == [1, 2]

   In a standalone script, put pool creation under the usual ``if __name__ == "__main__"`` guard on platforms that start workers with spawn.

.. py:function:: create_directory(directory_path)

   Create parent directories as needed. Returns the input path if created. Returns ``None`` for an existing path or a creation failure.

   .. testcode::

      from pathlib import Path
      from tempfile import TemporaryDirectory

      from llmabba.utils.tools import create_directory

      with TemporaryDirectory() as directory:
          path = Path(directory) / "one" / "two"
          assert create_directory(path) == path
          assert create_directory(path) is None

.. py:function:: save_train_duration(file_name, test_duration)

   Write a one-row CSV with the column ``train_duration``. The numeric duration is passed as ``test_duration`` and stored without conversion.

.. py:function:: save_test_duration(file_name, test_duration)

   Write the same layout with a ``test_duration`` column.

   .. testcode::

      from pathlib import Path
      from tempfile import TemporaryDirectory

      import pandas as pd
      from llmabba.utils.tools import save_train_duration, save_test_duration

      with TemporaryDirectory() as directory:
          train_csv = Path(directory) / "train.csv"
          test_csv = Path(directory) / "test.csv"
          save_train_duration(train_csv, 12.5)
          save_test_duration(test_csv, 0.9)
          assert pd.read_csv(train_csv).loc[0, "train_duration"] == 12.5
          assert pd.read_csv(test_csv).loc[0, "test_duration"] == 0.9

External baselines
------------------

.. py:currentmodule:: llmabba.utils.regressor_tools

.. py:function:: create_regressor(regressor_name, input_shape, output_directory, verbose=1, itr=1)

   Dispatch to the external experiment package ``models``. Recognized names include ``"lr"``, ``"ridge"``, ``"svr"``, ``"random_forest"``, ``"xgboost"``, ``"fcn"``, ``"resnet"``, ``"inception"``, and ``"rocket"``. Returns the selected backend or ``None`` for an unknown name. Install the matching experiment modules before using a recognized name.

.. py:function:: fit_regressor(output_directory, regressor_name, X_train, y_train, X_val=None, y_val=None, itr=1)

   Construct and fit an external baseline. Derives the per-sample input shape from ``X_train``. Pass validation data only to a backend that supports it. Returns the fitted backend. In this wrapper ``itr`` is passed into the factory's fourth positional slot, which is ``verbose``.

   .. code-block:: python

      import numpy as np
      from llmabba.utils.regressor_tools import create_regressor, fit_regressor

      # Requires the original experiment package "models".
      X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
      y = np.array([1.0, 2.0, 3.0])
      baseline = create_regressor("ridge", X.shape[1:], "results/")
      fitted = fit_regressor("results/", "ridge", X, y)

.. py:currentmodule:: llmabba.utils.transformer_tools

.. py:function:: create_transformer(transformer_name, **kwargs)

   Construct an external ``transform.transformers`` backend. ``"pca"`` selects PCA, ``"fpca"`` and ``"fpca_bspline"`` select FPCA, and other names delegate to the generic transformer. Extra keyword arguments go to the backend constructor.

.. py:function:: fit_transformer(transformer_name, X_train, flatten=False, **kwargs)

   Construct and fit a transformer and return ``(transformed_values, fitted_transformer)``. With ``flatten=True``, reshape a three-dimensional training batch into a single channel before fitting. The external transformer package must be importable.

   .. code-block:: python

      import numpy as np
      from llmabba.utils.transformer_tools import create_transformer, fit_transformer

      # Requires the original experiment package "transform".
      signals = np.arange(24.0).reshape(4, 3, 2)
      transformer = create_transformer("pca")
      transformed, fitted = fit_transformer("pca", signals, flatten=True)
