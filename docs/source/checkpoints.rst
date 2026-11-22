Save, reload, and move a run
============================

Files produced during training
------------------------------

:meth:`llmabba.LLMABBA.process` stores the fitted scaler and codebook using paths relative to your working directory.

.. code-block:: text

   ../save/<project_name>/<task>_Scaler_save.pkl
   ../save/<project_name>/<task>_ABBA_save.pkl

:meth:`llmabba.LLMABBA.train` saves the LoRA checkpoint under ``output_dir``. Keep the selected checkpoint directory, both pickle files, and a record of the base model, task, prompt, and maximum length. Use a distinct project name and output directory for independent runs.

Reload in a new process
-----------------------

Start with the same working directory, or place the files so the relative preprocessing paths resolve. In the tutorial function below, ``checkpoint_dir`` is the training ``output_dir`` and ``data`` contains unseen samples or a forecast history in the fitted input format.

.. literalinclude:: ../examples/llm_workflow.py
   :language: python
   :pyobject: predict

``inference()`` loads the artifacts and transforms new values using the saved encoder. For a directly fitted ABBA codebook, see :doc:`reference/artifacts` for a standalone save-and-load example. Open pickle files only from trusted sources.
