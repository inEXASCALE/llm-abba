LLM-ABBA documentation
======================

LLM-ABBA represents numerical time series as sequences of symbols. It uses ABBA to approximate signals with piecewise linear segments and assign symbols to similar segments. The symbolic sequences can be reconstructed to approximate the original signals or passed to a language model for classification, regression, and forecasting.

Use :doc:`getting-started` to run symbolic approximation without downloading a language model. The :doc:`classification` tutorial starts with raw ECG samples and walks through fine-tuning Mistral-7B and measuring performance on held-out samples.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting-started
   classification
   workflows
   api
   references
   license
