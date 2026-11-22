Dataset loading and regression metrics
======================================

.. py:currentmodule:: llmabba.utils.data_loader

.. py:exception:: TsFileParseException

   Raised by the loader for malformed data or unsupported metadata. The loader example below shows the format needed for a regression dataset and how to handle a parsing error.

.. py:function:: load_from_tsfile_to_dataframe(full_file_path_and_name, return_separate_X_and_y=True, replace_missing_vals_with='NaN')

   Read a UEA-style ``.ts`` file. For the numeric ``@targetlabel true`` format used in :doc:`../regression`, every row has one or more comma-separated channels, then a colon and the target.

   :param full_file_path_and_name: Local path to the dataset file.
   :param return_separate_X_and_y: Return targets separately when available. With false, targets are appended to the frame as ``class_vals``.
   :param replace_missing_vals_with: Text used in place of ``?`` before numeric conversion.
   :returns: A nested pandas DataFrame and target array when returning labels separately, or a DataFrame alone. Each feature cell contains one channel as a pandas Series.
   :raises TsFileParseException: Required metadata is missing or the rows have inconsistent dimensions.

   .. testcode:: data

      from pathlib import Path
      from tempfile import TemporaryDirectory

      from llmabba.utils.data_loader import (
          TsFileParseException, load_from_tsfile_to_dataframe,
      )

      with TemporaryDirectory() as directory:
          path = Path(directory) / "toy.ts"
          path.write_text(
              "@problemname toy\n@timestamps false\n@univariate false\n"
              "@targetlabel true\n@data\n1,2,3:4,5:62.5\n",
              encoding="utf-8",
          )
          frame, targets = load_from_tsfile_to_dataframe(path)
          assert frame.shape == (1, 2)
          assert float(targets[0]) == 62.5

          path.write_text("@problemname toy\n@timestamps maybe\n", encoding="utf-8")
          try:
              load_from_tsfile_to_dataframe(path)
          except TsFileParseException:
              pass
          else:
              raise AssertionError("An invalid timestamps value should fail")

.. py:currentmodule:: llmabba.utils.regressor_tools

.. py:function:: process_data(X, min_len, scaler=None)

   Convert per-channel pandas Series in a nested frame to a dense batch. Interpolate missing entries within a channel where possible, align shorter channels to the longest channel of the same sample, and crop or edge-pad each sample to ``min_len``.

   :param X: Nonempty frame with a series in each cell.
   :param min_len: Positive number of output time steps.
   :param scaler: Optional scaler. When supplied, this call fits it to all supplied time steps. For a held-out set, transform using a scaler fitted on training data.
   :returns: ``(values, scaler)`` where values have shape ``(samples, min_len, features)``.

   .. testcode:: data

      from llmabba.utils.regressor_tools import process_data

      values, fitted_scaler = process_data(frame, min_len=3)
      assert values.shape == (1, 3, 2)
      assert fitted_scaler is None

.. py:function:: calculate_regression_metrics(y_true, y_pred, y_true_val=None, y_pred_val=None)

   Compare aligned numeric predictions with targets. Returns a one-row pandas DataFrame with ``rmse`` and ``mae``. Passing a validation target array requires validation predictions and adds ``rmse_val`` and ``mae_val``.

   .. testcode:: data

      from llmabba.utils.regressor_tools import calculate_regression_metrics

      metrics = calculate_regression_metrics([60.0, 72.0], [62.0, 70.0])
      assert metrics.loc[0, "rmse"] == metrics.loc[0, "mae"] == 2.0
