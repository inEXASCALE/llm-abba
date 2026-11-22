PTBDB classification with Mistral-7B
====================================

The PTBDB example fine-tunes `Mistral-7B-Instruct-v0.1 <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1>`_ to classify ECG samples. In this example, ``0`` means abnormal and ``1`` means normal.

Run the demo
------------

Install the source checkout and follow the CUDA setup in :doc:`getting-started`. Run the command from the repository root.

.. code-block:: bash

   cd examples
   WANDB_MODE=disabled python demo-ts-classification.py

Use ``--model /path/to/model`` to load the same model from a local snapshot. The demo uses 4-bit NF4 weights, LoRA, a maximum sequence length of 424 tokens, and a training batch size of 4 per device. It also opens an ABBA reconstruction plot during preprocessing. Close the plot to continue, or omit ``draw_ts=True`` for a headless run.

Prepare the data and train
--------------------------

1. Load the normal and abnormal CSV files from ``test_data/``. Exclude the last column of each CSV from the signals, since it contains the dataset label. The example constructs its own labels from the source files.
2. Make a stratified 80/20 split of the original samples with ``random_state=42``. It holds out 2,911 test samples, then selects a stratified subset of 500 samples from the remaining rows for development.
3. Pass the development subset to ``process()``. It creates an internal 80/20 train/validation split, fits the scaler and ABBA codebook on the training rows, and uses that codebook to transform both splits. The default inner split is random and unstratified. Seed NumPy before calling ``process()`` if you need to reproduce that split.
4. Train for one epoch with ``balance_classes=True``. The sampler resamples the tokenized training rows to approximately equal class counts at the original epoch size. The validation set retains its original class distribution. Set ``sampling_seed`` to control the resampling.
5. Reload the saved adapter and use the original prompt for inference. The classifier uses the tokenizer's chat template when available, masks prompt and padding tokens in the training loss, and restricts decoding to the saved class labels.

The example uses ``abba_alpha=0.2`` for codebook aggregation. Select aggregation settings, sampling options, and any decision thresholds on training and validation data before evaluating the held-out test set.

Evaluate the held-out set
-------------------------

The demo prints one prediction. To evaluate all held-out rows, replace the inference block at the end of the script with the following code. Keep ``test_data`` as the full array from the first split. Releasing the training model before reloading the adapter reduces GPU memory use when both stages run in one process.

.. code-block:: python

   import gc

   import torch
   from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

   del model_input
   gc.collect()
   torch.cuda.empty_cache()

   checkpoint = get_last_checkpoint(output_dir)
   if checkpoint is None:
       raise FileNotFoundError(f"No adapter checkpoint found in {output_dir}")

   model, tokenizer = LLMABBA_classification.model(
       peft_file=checkpoint,
       model_name=model_name,
       max_len=424,
   )
   predictions = np.asarray(
       LLMABBA_classification.inference(
           project_name=project_name,
           data=test_data,
           task=task_type,
           prompt=prompt_input,
           ft_model=model,
           model_tokenizer=tokenizer,
       ),
       dtype=str,
   )
   labels = ["0", "1"]
   actual = test_target.astype(str)
   valid = np.isin(predictions, labels)
   majority_label = str(np.bincount(train_target).argmax())

   print("Accuracy", accuracy_score(actual, predictions))
   print("Recall [abnormal, normal]", recall_score(
       actual, predictions, labels=labels, average=None, zero_division=0,
   ))
   print("Confusion matrix (valid predictions only)", confusion_matrix(
       actual[valid], predictions[valid], labels=labels,
   ), sep="\n")
   print("Invalid predictions", np.count_nonzero(~valid))
   print("Majority-class baseline", np.mean(actual == majority_label))

Rows of the confusion matrix are actual labels and columns are predicted labels, in the order abnormal, normal. Accuracy and recall count invalid predictions as misses. The majority-class baseline predicts the most frequent class in the development subset for every test sample.

Complete example
----------------

.. literalinclude:: ../../examples/demo-ts-classification.py
   :language: python
   :caption: examples/demo-ts-classification.py
