Getting started
===============

Install from PyPI
-----------------

Create and activate a virtual environment in a POSIX shell.

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install llmabba

With Conda, create an environment with ``conda create -n llmabba python=3.12 pip``, activate it with ``conda activate llmabba``, and run the same pip command.

Install the source checkout
---------------------------

The tutorials use the scripts and datasets in the repository. Install the matching checkout to follow them.

.. code-block:: bash

   git clone https://github.com/inEXASCALE/llm-abba.git
   cd llm-abba
   python -m pip install .

A source build can compile the Cython extensions for segmentation, aggregation, and reconstruction. The package also supplies Python implementations.

Try ABBA on the CPU
-------------------

.. testcode::

   from llmabba import ABBA

   encoder = ABBA(tol=0.1, alpha=0.1, verbose=0)
   symbols = encoder.encode([[1.0, 1.5, 2.0], [2.0, 3.0, 4.0]])
   reconstructed = encoder.decode(symbols)
   print(len(symbols), len(reconstructed))

.. testoutput::

   2 2

The input contains two equal-length series, one per row. The encoder learns a shared codebook. The decoder returns an approximate reconstruction for each row. See :doc:`symbolic` for transforming additional rows and inspecting the compression.

Set up a GPU for fine-tuning
----------------------------

The Mistral-7B tutorials use CUDA-enabled PyTorch and bitsandbytes NF4 quantization. Refer to the `PyTorch installation selector <https://pytorch.org/get-started/locally/>`_ and the `bitsandbytes installation guide <https://huggingface.co/docs/bitsandbytes/en/installation>`_ for compatible builds.

.. code-block:: bash

   nvidia-smi
   python -c "import torch, bitsandbytes; print(torch.version.cuda, torch.cuda.is_available(), bitsandbytes.__version__)"

LLM-ABBA uses the same 4-bit compute dtype when loading a model for training and inference. It selects BF16 where CUDA supports it natively and FP16 on other CUDA devices, including the RTX 2080 Ti. Choose ``quant_process=False``, or ``--no-quantization`` in the tutorial runner, when you have a model and device that can run without bitsandbytes.

Choose a tutorial
-----------------

* :doc:`classification` predicts one label per ECG recording.
* :doc:`regression` estimates one numeric target per PPG recording.
* :doc:`forecasting` predicts a future window from a multivariate history.
