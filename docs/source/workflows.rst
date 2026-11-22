Regression, forecasting, and checkpoints
========================================

The :class:`~llmabba.llm_abba.LLMABBA` workflow loads a base model and tokenizer with ``model()``, builds tokenized training and validation datasets with ``process()``, fine-tunes a LoRA adapter with ``train()``, and predicts with ``inference()``. The :doc:`classification` tutorial applies the same workflow to discrete labels.

Regression
----------

Pass a dictionary with ``X_data`` and ``Y_data`` to ``process(task="regression", ...)``. Use a numeric array of shape ``(samples, time)`` for univariate batches or ``(samples, time, features)`` for multivariate batches. Provide one numeric target per sample in ``Y_data``. During preprocessing, the scaler and ABBA codebook are fitted on the training portion of the input.

At inference, use the same time and feature dimensions as training for multivariate samples. The return value is a string for a single prediction or a list of strings in input order for a batch. Ask the model for a numeric answer in the prompt and parse each response before computing regression metrics. Track responses that cannot be parsed as finite numbers.

Forecasting
-----------

Pass a chronological array of shape ``(time, features)`` to ``process(task="forecasting", ...)``. Set ``seq_len_pre`` to the number of history rows and ``seq_len_post`` to the forecast horizon. The series must have at least ``2 * (seq_len_pre + seq_len_post)`` rows. Windows can overlap within a split. Training and validation windows come from separate time ranges, with the scaler fitted on the training range.

At inference, pass one history window of shape ``(seq_len_pre, features)`` with the same feature order. The method decodes the predicted symbols with the saved ABBA codebook and returns an array of shape ``(seq_len_post, features)`` in the original units. It uses the stored horizon unless ``seq_len_post`` is supplied explicitly.

For evaluation, reserve a later time range before calling ``process()``. Compare the forecast with the next ``seq_len_post`` rows immediately after each history window. If generation fails with ``The generated forecast contains no recognized ABBA symbols``, check that the prompt requests the trained symbol format and that the adapter and codebook belong to the same run. Compare prompt or alphabet choices using validation windows.

Settings
--------

* Set ``abba_tol``, ``abba_alpha``, ``abba_init``, and ``abba_scl`` on ``LLMABBA`` to configure symbolic approximation. Set ``lora_r``, ``lora_alpha``, and ``lora_dropout`` to configure the adapter.
* Use ``scalar="z-score"`` or ``scalar="min-max"`` in ``process()``. With ``alphabet_set=None``, the method tries to build an alphabet from the tokenizer vocabulary. You can also pass a built-in alphabet selector or a list of nonempty, whitespace-free symbols.
* Set ``num_epochs`` and ``output_dir`` in ``train()`` to control the training schedule and checkpoint location.
* For regression and forecasting, set ``llm_max_new_tokens`` to bound the generated continuation. Use ``llm_temperature=0.0`` for greedy decoding or a positive value to sample. ``llm_repetition_penalty`` controls how strongly repeated tokens are penalized.
* Set ``draw_ts=True`` in ``process()`` to compare an input with its symbolic reconstruction. For forecasting, the same option in ``inference()`` plots the history and forecast.

Save and reload a run
---------------------

``process()`` saves two preprocessing artifacts relative to the current working directory.

.. code-block:: text

   ../save/<project_name>/<task>_Scaler_save.pkl
   ../save/<project_name>/<task>_ABBA_save.pkl

``train()`` writes adapter checkpoints under its ``output_dir``. Keep an adapter together with the matching scaler, codebook, base model identifier, and prompt. When starting a new process for inference, use the same working directory and project name, or arrange the preprocessing files at the expected paths. Load the adapter with ``model(peft_file=checkpoint_path, model_name=base_model, max_len=max_len)``, then pass the returned model and tokenizer to ``inference()``.

Inference loads the existing scaler and codebook and applies them to the new data. If those files are missing, restore the matching artifacts or run preprocessing on the training data before fine-tuning. The helpers :func:`~llmabba.llm_abba.save_abba` and :func:`~llmabba.llm_abba.load_abba` write and read pickle files. Load these files only from trusted sources.
