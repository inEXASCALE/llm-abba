Multivariate forecasting with ETTh1
===================================

Predict the next 24 hourly observations of ``HUFL`` and ``OT`` from a 24-hour history. The output has the same feature order and units as the source columns.

Prepare chronological windows
-----------------------------

Sort by timestamp and keep the first 14 days as development data. Hold out the next 24 hours as the target. The final 24 hours of the development interval are observable history at forecast time. Inside ``process()``, the rows used to build training and validation windows come from separate chronological ranges. The scaler fits on the earlier range.

.. literalinclude:: ../examples/llm_workflow.py
   :language: python
   :pyobject: forecasting_data

Train and reload
----------------

Use the environment in :doc:`getting-started`. The runner passes ``seq_len_pre=24`` and ``seq_len_post=24`` to ``process()`` and saves the horizon with the codebook.

.. code-block:: bash

   cd examples
   WANDB_MODE=disabled python ../docs/examples/llm_workflow.py forecasting train
   WANDB_MODE=disabled python ../docs/examples/llm_workflow.py forecasting evaluate

Both stages use the same base model, project name, and prompt. Supply ``--model /path/to/model`` to both commands to use a local model. The evaluation stage passes a single ``(24, 2)`` history to ``inference()``. It decodes the generated symbols and returns a ``(24, 2)`` forecast. If the model generates no recognized ABBA symbols, see :doc:`troubleshooting`.

Evaluate the horizon
--------------------

The runner prints a separate MAE for each feature. Its persistence baseline repeats the final observed row throughout the forecast window. The example evaluates one horizon. To benchmark across time, choose several origins in a reserved test range and report how forecasts aggregate across origins and features.

.. literalinclude:: ../examples/llm_workflow.py
   :language: python
   :pyobject: evaluate_forecasting
