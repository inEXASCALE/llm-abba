API reference
=============

Symbolic approximation
----------------------

.. autoclass:: llmabba.ABBA

   The ``ABBA`` interface fits a shared codebook and reconstructs numerical time series from their symbols. See :doc:`getting-started` for parameter descriptions and examples.

   .. automethod:: encode

      Learn a codebook from the input and return the symbol sequences.

   .. automethod:: decode

      Reconstruct the sequences returned by ``encode()`` using the saved starting values.

LLM workflows
-------------

.. autoclass:: llmabba.llm_abba.LLMABBA

   .. automethod:: model(peft_file=None, model_name=None, max_len=2048)

      Return the model and tokenizer. Omit ``peft_file`` to initialize a LoRA adapter for training, or pass a saved adapter checkpoint for inference.

   .. automethod:: process
   .. automethod:: train
   .. automethod:: inference

Preprocessing artifacts
-----------------------

.. autofunction:: llmabba.llm_abba.save_abba

   Serialize a fitted ABBA codebook or scaler to a pickle file.

.. autofunction:: llmabba.llm_abba.load_abba

   Load an artifact from a trusted pickle file.
