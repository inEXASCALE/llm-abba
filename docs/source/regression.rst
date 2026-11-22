Time-series regression with IEEEPPG
===================================

Predict heart rate in beats per minute from a window of PPG and acceleration signals. Each case has several input channels and one numeric target.

Prepare the data
----------------

Keep the provided IEEEPPG training and test files separate. :func:`llmabba.utils.data_loader.load_from_tsfile_to_dataframe` returns a nested frame and target array. To keep the tutorial short, use 500 training-file cases and 64 time steps per input. Apply the same length to data from the test file. :func:`llmabba.utils.regressor_tools.process_data` returns arrays in ``(samples, time, features)`` order. The scaler and codebook used by the model are fitted inside ``process()`` using the training portion of the development data.

.. literalinclude:: ../examples/llm_workflow.py
   :language: python
   :pyobject: regression_data

Train and reload
----------------

Use the environment in :doc:`getting-started`. Run both commands from the same working directory.

.. code-block:: bash

   cd examples
   WANDB_MODE=disabled python ../docs/examples/llm_workflow.py regression train
   WANDB_MODE=disabled python ../docs/examples/llm_workflow.py regression evaluate

The runner uses Mistral-7B in 4-bit NF4 with a 424-token budget and trains for one epoch. It saves artifacts under ``../save/regression-tutorial`` and reloads the adapter in the evaluation process. Pass ``--model /path/to/model`` to both commands to use a local model. The default evaluation uses the full test file. Add ``--eval-limit 8`` for a quick reload and generation check.

The prompt asks for one number. For generation, the example loads the same tokenizer vocabulary with ``add_eos_token=False`` so the response follows the results marker. This setting applies to the tutorial's inference prompt. Training examples retain their usual end-of-sequence token.

Evaluate numeric responses
--------------------------

The API returns generated text. Convert each response to a finite float and report how many responses are valid. Calculate RMSE and MAE on those rows and compare with the development-mean baseline evaluated on the same rows. Interpret the errors together with the response coverage. The short signal window and development subset make this a workflow example.

.. literalinclude:: ../examples/llm_workflow.py
   :language: python
   :pyobject: evaluate_regression
