LLMABBA
=======

.. py:currentmodule:: llmabba

.. autoclass:: llmabba.LLMABBA

   Configure symbolic preprocessing and a LoRA adapter for a causal language model. See :doc:`../workflows` for the call order.

   .. list-table::
      :header-rows: 1

      * - Arguments
        - Meaning
      * - ``abba_tol``, ``abba_init``, ``abba_alpha``, ``abba_scl``, ``abba_verbose``
        - Segmentation, grouping, length weighting, and verbosity as in :class:`llmabba.XABBA`.
      * - ``bits_for_len``, ``bits_for_inc``
        - Precision of ABBA prototype lengths and end values.
      * - ``lora_r``, ``lora_alpha``, ``lora_dropout``
        - Adapter rank, scaling, and dropout.
      * - ``target_modules``, ``modules_to_save``
        - Architecture-specific module names adapted by LoRA and modules stored alongside the adapter. With ``target_modules=None``, the defaults include the attention projections, gated feed-forward projections, and ``lm_head``. The default saved module is ``embed_tokens``.
      * - ``quant_process``
        - Load the base model in bitsandbytes NF4 when true. Training and inference share dtype selection.

   .. testcode:: llm

      from llmabba import LLMABBA

      workflow = LLMABBA(abba_alpha=0.2, quant_process=False)

   .. automethod:: model(peft_file=None, model_name=None, max_len=2048)

      :param model_name: Model repository ID or local base model directory. The tokenizer must be loadable from the same identifier.
      :param peft_file: Optional saved LoRA checkpoint directory. Omit it to initialize an adapter for training.
      :param max_len: Token budget used to build training examples. For classification it must accommodate the instruction, longest label, and EOS.
      :returns: ``(model, tokenizer)``. The method also sets the instance's model name, tokenizer, and length budget.

      Example with a CUDA-enabled NF4 environment. To reload, pass your checkpoint directory as ``peft_file`` and reuse the base model and budget.

      .. code-block:: python

         from llmabba import LLMABBA

         workflow = LLMABBA(abba_alpha=0.2)
         model, tokenizer = workflow.model(
             model_name="mistralai/Mistral-7B-Instruct-v0.1", max_len=424,
         )

      The :doc:`../classification` tutorial runs both calls from the command line.

   .. py:method:: process(project_name, data, task, prompt, model_tokenizer, seq_len_pre=24, scalar='z-score', seq_len_post=24, alphabet_set=None, draw_ts=False)

      Build training and validation datasets and save fitted preprocessing artifacts. Fit the scaler and codebook from the training partition only.

      :param project_name: Name of the preprocessing directory under ``../save``.
      :param data: Supervised dictionary with ``X_data`` and ``Y_data``, or a chronological array for forecasting. See :doc:`../input-data` for shapes.
      :param task: ``"classification"``, ``"regression"``, or ``"forecasting"``.
      :param prompt: Instruction to use with the symbolic sequence. Reuse it at inference.
      :param model_tokenizer: Tokenizer returned by :meth:`model` or a compatible tokenizer with padding and EOS configured for classification.
      :param seq_len_pre: History length for forecasting.
      :param seq_len_post: Horizon for forecasting. The two forecast lengths must be positive and the array must contain at least twice their sum in rows.
      :param scalar: ``"z-score"`` or ``"min-max"``.
      :param alphabet_set: Integer selector, list of nonempty whitespace-free symbols, or ``None`` to derive symbols from the tokenizer vocabulary.
      :param draw_ts: Display a training input and its reconstruction.
      :returns: ``(train_dataset, val_dataset)`` as tokenized Hugging Face datasets.
      :raises ValueError: Unsupported task or scaler, unsuitable inputs, or classification labels that cannot be encoded distinctly.

      The following example uses a local tokenizer and a temporary working directory. It exercises preprocessing without loading model weights. The next two methods reuse this setup.

      .. testcode:: llm

         from contextlib import chdir
         from pathlib import Path
         from tempfile import TemporaryDirectory

         import numpy as np
         from tokenizers import Tokenizer
         from tokenizers.models import WordLevel
         from tokenizers.pre_tokenizers import Whitespace
         from transformers import PreTrainedTokenizerFast

         vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "A": 3, "2": 4}
         backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
         backend.pre_tokenizer = Whitespace()
         tokenizer = PreTrainedTokenizerFast(
             tokenizer_object=backend, pad_token="<pad>",
             eos_token="<eos>", unk_token="<unk>",
         )
         workflow.max_len = 32
         inputs = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]] * 4)
         targets = np.array([2.0, 4.0] * 4)
         with TemporaryDirectory() as directory:
             run = Path(directory) / "run"
             run.mkdir()
             with chdir(run):
                 train_set, val_set = workflow.process(
                     "toy", {"X_data": inputs, "Y_data": targets},
                     "regression", "Return a number.", tokenizer, alphabet_set=0,
                 )
                 assert len(train_set) + len(val_set) == len(inputs)
                 assert Path("../save/toy/regression_ABBA_save.pkl").is_file()

   .. py:method:: tokenize(prompt_input)

      Tokenize a text prompt to the current budget with truncation and padding. Configure the tokenizer with :meth:`model` or :meth:`process` first. Returns a tokenizer mapping with ``input_ids``, ``attention_mask``, and ``labels``. Padding positions in ``labels`` are masked with ``-100``.

      .. testcode:: llm

         encoded = workflow.tokenize("A 2")
         assert len(encoded["input_ids"]) == workflow.max_len
         assert encoded["labels"][-1] == -100

   .. py:method:: build(data_point)

      Convert a mapping with ``text_inputs`` (space-separated symbols) and ``text_outputs`` (target) into a training example. Use the instance after :meth:`process` has initialized its prompt and tokenizer. Returns token IDs, attention mask, and labels. A processed classifier masks the prompt and supervises label and EOS tokens. Other tasks mask padding.

      .. testcode:: llm

         row = workflow.build({"text_inputs": "A", "text_outputs": 2})
         assert len(row["labels"]) == workflow.max_len

   .. py:method:: train(model_input, num_epochs, output_dir, train_dataset, val_dataset, *, balance_classes=False, sampling_seed=42)

      Fine-tune the adapter returned by :meth:`model` using the datasets from :meth:`process`. The Trainer evaluates and saves by epoch. This call returns ``None`` and writes checkpoints and logs.

      :param model_input: Causal model with the trainable adapter.
      :param num_epochs: Number of training epochs.
      :param output_dir: Directory for Trainer checkpoints.
      :param train_dataset: Tokenized training dataset.
      :param val_dataset: Tokenized validation dataset.
      :param balance_classes: Resample the training examples to roughly equal class counts while preserving the epoch size.
      :param sampling_seed: Seed for balanced sampling.
      :raises ValueError: Balanced sampling was requested for a task other than classification.

      This function in the tutorial runner loads the model, prepares the data and calls ``train()``. The task tutorials supply a runnable command and dataset.

      .. literalinclude:: ../../examples/llm_workflow.py
         :language: python
         :pyobject: train_adapter

   .. py:method:: inference(project_name, data, task, prompt, ft_model, model_tokenizer, scalar='z-score', seq_len_pre=24, seq_len_post=None, llm_max_length=1024, llm_repetition_penalty=1.7, llm_temperature=0.0, llm_max_new_tokens=256, draw_ts=False)

      Use the saved scaler and codebook with a fine-tuned model. The project, task, prompt, tokenizer, and adapter must match the training run.

      :param project_name: Name passed to :meth:`process`.
      :param data: Unseen sample, supervised batch, or one forecasting history, per :doc:`../input-data`.
      :param task: Training task.
      :param prompt: Original task instruction.
      :param ft_model: Base model with the saved adapter loaded.
      :param model_tokenizer: Tokenizer returned when reloading the model.
      :param scalar: Compatibility argument. Inference uses the saved scaler.
      :param seq_len_pre: Expected history length when a forecasting artifact lacks fitted shape information.
      :param seq_len_post: Requested forecast horizon. Omit to use the saved horizon when available.
      :param llm_max_length: Total generation length for regression and forecasting if ``llm_max_new_tokens=None``.
      :param llm_max_new_tokens: Continuation limit for regression and forecasting.
      :param llm_repetition_penalty: Repetition penalty for free-form generation.
      :param llm_temperature: Zero for greedy generation. Positive values sample.
      :param draw_ts: Show the history and forecast for the forecasting task.
      :returns: A string for a single supervised input, a list of strings for multiple supervised inputs, or a numerical array for a forecast.
      :raises FileNotFoundError: A matching preprocessing artifact is missing.
      :raises ValueError: Unknown task, missing prompt, incompatible forecast history, or no recognizable forecast symbols.

      The complete runner uses a second process for adapter reload and prediction.

      .. literalinclude:: ../../examples/llm_workflow.py
         :language: python
         :pyobject: predict
