Getting started
===============

Installation
------------

Create and activate a virtual environment, then install the latest published version from PyPI. The commands below use a POSIX shell.

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install llmabba

You can also use Conda. Create an environment with ``conda create -n llmabba python=3.12 pip``, activate it with ``conda activate llmabba``, and run the same pip command.

To use the examples and APIs from this documentation, install the corresponding source checkout. The repository includes the scripts and their datasets.

.. code-block:: bash

   git clone https://github.com/inEXASCALE/llm-abba.git
   cd llm-abba
   python -m pip install .

Source builds can use a C compiler to build Cython extensions for compression, symbol assignment, and reconstruction. The package also supplies Python implementations for environments without the compiled extensions.

GPU setup for language-model tasks
----------------------------------

``ABBA`` runs on the CPU. The Mistral-7B example uses a CUDA-enabled PyTorch build and a bitsandbytes release that supports 4-bit NF4 on your GPU. Follow the `PyTorch installation selector <https://pytorch.org/get-started/locally/>`_ and the `bitsandbytes installation guide <https://huggingface.co/docs/bitsandbytes/en/installation>`_ for your platform before running the demo.

Verify that PyTorch sees the GPU and bitsandbytes can be imported.

.. code-block:: bash

   python -c "import torch, bitsandbytes; print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name() if torch.cuda.is_available() else 'unavailable'); print('bitsandbytes:', bitsandbytes.__version__)"

Four-bit training and inference use the same compute dtype selection. They use BF16 when the CUDA GPU supports it natively and FP16 on other CUDA GPUs, including the RTX 2080 Ti. ``LLMABBA(quant_process=False)`` disables bitsandbytes quantization if the base model fits without it.

Symbolic approximation and reconstruction
-----------------------------------------

``ABBA`` uses fixed-point adaptive piecewise linear continuous approximation (FAPCA). It approximates the signal with segments, groups segments using their lengths and value increments, then assigns a symbol to each group. Decoding uses the stored starting value and segment prototypes to reconstruct an approximation.

.. code-block:: python

   from llmabba import ABBA

   series = [
       [1.2, 1.4, 1.3, 1.8, 2.2, 2.4, 2.1],
       [1.2, 1.3, 1.2, 2.2, 1.4, 2.4, 2.1],
   ]
   abba = ABBA(tol=0.1, alpha=0.1, init="agg", scl=3)
   symbols = abba.encode(series)
   reconstruction = abba.decode(symbols)
   for sequence, values in zip(symbols, reconstruction):
       print(" ".join(sequence), values)

For a batch of univariate series, pass a two-dimensional array or a list of equal-length rows. ``encode()`` fits a shared codebook and returns a list of symbol sequences. ``decode()`` reconstructs them using the starting values stored by that call. To encode more samples with an existing codebook, call ``symbols, starts = abba.transform(new_series)`` and reconstruct them with ``abba.inverse_transform(symbols, start_set=starts)``.

* ``tol`` controls the error allowed when forming a segment. Smaller tolerances typically produce more segments.
* With ``init="agg"``, ``alpha`` is the aggregation tolerance for grouping segments. Larger values typically produce a smaller symbol vocabulary.
* ``init`` selects aggregation (``"agg"``), sampled k-means (``"f-kmeans"``), or standard k-means (``"kmeans"``).
* ``scl`` weights the segment length relative to the increment during symbol assignment.
* ``bits_for_len`` and ``bits_for_inc`` control the quantization precision of the codebook prototypes.

Compare symbolic sequence length, codebook size, and reconstruction error when choosing settings for compression. For a downstream task, compare settings on validation data with the metric you intend to report. See the `fABBA project <https://github.com/nla-group/fABBA>`_ and :doc:`references` for algorithm details.
