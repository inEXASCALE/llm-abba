Language-model workflow
=======================

The :class:`llmabba.LLMABBA` interface connects a causal language model to a fitted symbolic encoder. Choose a base model whose module names match the LoRA configuration.

1. Call :meth:`llmabba.LLMABBA.model` to load a model and tokenizer and initialize an adapter. Set ``max_len`` to the token budget for your prompt, symbols, and answer.
2. Call :meth:`llmabba.LLMABBA.process` to build the training and validation datasets. Keep the validation split and saved preprocessing artifacts together.
3. Call :meth:`llmabba.LLMABBA.train` with the resulting datasets and an explicit checkpoint directory.
4. In a fresh process, call ``model(peft_file=..., model_name=...)`` with the same base model and token budget. Pass its return values to :meth:`llmabba.LLMABBA.inference` with the project name and original prompt.

The tutorials provide the data and evaluation code for :doc:`classification`, :doc:`regression`, and :doc:`forecasting`. They share the :download:`llm_workflow.py <../examples/llm_workflow.py>` runner at ``docs/examples/llm_workflow.py`` in the checkout. See :doc:`input-data` for shapes and :doc:`checkpoints` for the exact artifact paths.

Prompt and symbol settings
--------------------------

Pass ``alphabet_set=-1`` for the lower-case-first built-in symbol order, or give ``process()`` a list of nonempty symbols without whitespace. By default, ``process()`` tries to derive an alphabet from the tokenizer vocabulary. Inspect tokenized examples before training, especially when one ABBA symbol can map to several language-model tokens.

Classification uses the chat template when available, reserves space for the label and EOS, and supervises the answer tokens. Its decoder uses the label vocabulary stored in the codebook. Regression and forecasting generate a continuation and support ``llm_temperature``, ``llm_max_new_tokens``, and ``llm_repetition_penalty``. Keep enough space for the target when choosing the token budget for those tasks.
