LLM-ABBA documentation
======================

LLM-ABBA is a Python library for symbolic time-series analysis with large language models. It uses ABBA to approximate a signal with variable-length linear segments and assign symbols to groups of segments with similar lengths and changes in value. The resulting sequences can be decoded back into numerical approximations or used as language-model inputs for classification, regression, and forecasting.

The guides are written for researchers comparing time-series representations, practitioners fine-tuning models on numerical signals, and learners exploring how these techniques work together.

* :doc:`getting-started` shows how to encode and reconstruct a signal on the CPU without downloading a language model.
* :doc:`classification` starts with raw ECG samples and walks through fine-tuning Mistral-7B and measuring performance on held-out samples.
* :doc:`workflows` covers regression and forecasting input shapes, generation settings, and how to reuse a trained adapter.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting-started
   classification
   workflows
   api
   references
   license
