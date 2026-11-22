PTBDB classification with Mistral-7B
====================================

This tutorial fine-tunes `Mistral-7B-Instruct-v0.1 <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1>`_ to assign one class to an ECG recording. The tutorial uses ``0`` for abnormal and ``1`` for normal.

Prepare the data
----------------

The final column of each CSV is a dataset label. Drop it from the feature array and assign the tutorial labels by source file. Make a stratified 80/20 split of the raw rows, then select a stratified development subset of 500 from the remaining data. The held-out split has 2,911 rows.

.. literalinclude:: ../examples/llm_workflow.py
   :language: python
   :pyobject: classification_data

Train an adapter
----------------

Install the checkout and configure CUDA as described in :doc:`getting-started`. Start in the repository root.

.. code-block:: bash

   cd examples
   WANDB_MODE=disabled python ../docs/examples/llm_workflow.py classification train

The runner uses 4-bit NF4, one epoch, and a 424-token budget. ``process()`` splits the development subset into 400 training and 100 validation rows after the outer stratified split. It fits the scaler and codebook on training rows and transforms both splits. ``balance_classes=True`` resamples the training dataset to approximately equal class counts at the original epoch size. The runner seeds the random number generators with 42.

The adapter is saved under ``../save/classification-tutorial/checkpoints``. The codebook and scaler are saved in the corresponding project directory. See :doc:`checkpoints` for details. You can supply ``--model /path/to/model`` on both commands to use a local snapshot.

Evaluate the held-out set
-------------------------

Run evaluation as a separate process from the same ``examples`` directory.

.. code-block:: bash

   WANDB_MODE=disabled python ../docs/examples/llm_workflow.py classification evaluate

The runner reads the saved artifacts and uses the original prompt. The classifier uses the tokenizer's chat template when available, supervises the answer tokens, and restricts decoding to the saved label vocabulary. For a quick check, add ``--eval-limit 4``. Omit it for metrics on all 2,911 rows.

.. literalinclude:: ../examples/llm_workflow.py
   :language: python
   :pyobject: evaluate_classification

The confusion matrix orders actual labels by row and predicted labels by column, with abnormal first. Its entries include recognized labels only. Accuracy and recall count invalid outputs as misses. The majority baseline uses the most frequent class in the development subset. Choose any aggregation, sampling, or decision settings before evaluating the holdout.
