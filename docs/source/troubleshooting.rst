Troubleshooting
===============

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Symptom
     - What to check
   * - ``FileNotFoundError`` loading an adapter or preprocessor
     - Verify the checkpoint directory, project name, task, and working directory. The scaler and codebook live under ``../save`` relative to where you run inference.
   * - XABBA needs fitted parameters
     - Fit on the training input, or reload the fitted encoder before calling ``transform()``. Keep ``start_set`` aligned with the symbol sequences for reconstruction.
   * - A multivariate shape or forecast history is rejected
     - Match training time and feature dimensions, including feature order. For forecasting, pass one history window of the fitted length with shape ``(time, features)``.
   * - Classification always returns the same label
     - Check the majority-class baseline and recall by class on validation data. Try ``balance_classes=True`` for training and keep validation and test prevalence intact. Check the codebook and compare symbol settings on validation data.
   * - Forecast contains no recognized symbols
     - Use the prompt, tokenizer, and alphabet associated with the trained adapter. Inspect the generated symbol sequence and verify the codebook loaded from the same run.
   * - ``ShapeWarning`` during forecast reconstruction
     - The decoded segment lengths produced a different number of values than the requested horizon. XABBA pads short rows or truncates long ones during reshaping. Check the forecast shape and how often the adjustment occurs when evaluating the model.
   * - Regression outputs contain text or non-finite values
     - Ask for one numeric value, verify that the training target remains inside the token budget, and report the fraction of held-out outputs that parse as finite numbers.
   * - CUDA runs out of memory
     - Run training and evaluation in separate processes so each loads one base model. Reduce the token budget if prompts and targets still fit, or choose a smaller model.
   * - Training asks for Weights & Biases credentials
     - Follow the tutorial commands with ``WANDB_MODE=disabled`` for a local run, or configure W&B before training.
