LLM-ABBA
========


.. image:: https://img.shields.io/badge/Cython_Support-Accelerated-blue?style=flat&logoColor=cyan&labelColor=cyan&color=black
   :alt: Cython Support
   :target: https://github.com/inEXASCALE/llm-abba

.. image:: https://img.shields.io/github/license/inEXASCALE/llm-abba
   :alt: License
   :target: https://github.com/inEXASCALE/llm-abba/blob/main/LICENSE

`llmabba` is a software framework for time series analysis using Large Language Models (LLMs) based on symbolic representation, as introduced in the paper `LLM-ABBA: Symbolic Time Series Approximation using Large Language Models <https://arxiv.org/abs/2411.18506>`_.

Time series analysis involves identifying patterns, trends, and structures within data sequences. Traditional methods like discrete wavelet transforms or symbolic aggregate approximation (SAX) convert continuous time series into symbolic representations for better analysis and compression. However, these methods often struggle with complex patterns.

`llmabba` enhances these techniques by leveraging LLMs, which excel in pattern recognition and sequence prediction. By applying LLMs to symbolic time series, `llmabba` discovers rich, meaningful representations, offering:

- **Higher accuracy and compression**: Better symbolic representations via LLMs, improving data compression and pattern accuracy.
- **Adaptability**: Robust performance across domains like finance, healthcare, and environmental science.
- **Scalability**: Efficient handling of large-scale time series datasets.
- **Automatic feature discovery**: Uncovers novel patterns that traditional methods may miss.

Key Features
------------
- Symbolic Time Series Approximation: Converts time series into symbolic representations.
- LLM-Powered Encoding: Enhances compression and pattern discovery.
- Efficient and Scalable: Suitable for large-scale datasets.
- Flexible Integration: Compatible with machine learning and statistical workflows.

Installation
------------
To set up a virtual environment, use one of these methods:

**Using venv**:

.. code-block:: bash

   mkdir ~/.myenv
   python -m venv ~/.myenv
   source ~/.myenv/bin/activate

**Using conda**:

.. code-block:: bash

   conda create -n myenv
   conda activate myenv

Then, install `llmabba` via pip:

.. code-block:: bash

   pip install .


Basic Usage: Symbolic Approximation with ABBA
--------------------------------------------

The ``ABBA`` class provides quantized symbolic approximation using Fixed-point Adaptive Piecewise Linear Continuous Approximation (FAPCA). It converts numerical time series into symbolic strings (e.g., letters like 'A', 'B' representing trends such as increasing or decreasing segments) and supports reconstruction back to numerical form. 

**Key Parameters**:
- ``tol``: Tolerance for approximation error (smaller values increase segmentation granularity).
- ``alpha``: Quantization parameter (controls symbol assignment to segments).
- ``init``: Initialization method for FAPCA (e.g., 'agg' for aggregation-based).
- ``scl``: Scaling factor for approximation.

**Input Format**:
- A list of lists, where each inner list is a 1D time series of floating-point numbers (e.g., multiple time series samples).

**Output**:
- ``encode``: Returns a list of symbolic strings (e.g., ['A B C'] for each time series).
- ``decode``: Reconstructs the time series as a list of lists of floats, approximating the input.

**Example**:

.. code-block:: python

   from llmabba import ABBA

   # Sample time series data
   ts = [[1.2, 1.4, 1.3, 1.8, 2.2, 2.4, 2.1], [1.2, 1.3, 1.2, 2.2, 1.4, 2.4, 2.1]]
   abba = ABBA(tol=0.1, alpha=0.1, init='agg', scl=3)
   symbolic_representation = abba.encode(ts)
   print("Symbolic Representation:", symbolic_representation)
   reconstruction = abba.decode(symbolic_representation)
   print("Reconstruction:", reconstruction)

This example encodes two time series into symbolic strings and reconstructs them, with minor deviations based on ``tol``. The output can be used for pattern matching, compression, or visualization.

Advanced Usage: Time Series Tasks with LLMABBA
---------------------------------------------

The ``LLMABBA`` class integrates ABBA with LLMs (e.g., Mistral-7B) for advanced tasks like classification, regression, and forecasting. It processes time series into symbolic representations, tokenizes them for LLM input, fine-tunes the model, and performs inference. The class supports 4-bit quantization (via BitsAndBytesConfig) and LoRA (Low-Rank Adaptation) for efficient fine-tuning, along with FSDP for distributed training. This guide provides a overview of LLM-ABBA's functionality, including installation, symbolic approximation with the `ABBA` class, and advanced tasks (classification, regression, forecasting) using the ``LLMABBA`` class. We refer to the GitHub repository (https://github.com/inEXASCALE/llm-abba) and the `examples` folder for additional details and parameter tuning. For more details on the use of fABBA and ABBA, switch to GitHub repository (https://github.com/nla-group/fABBA). 


**Key Steps**:

* Step 1. **Data Preparation**: Load numerical time series and labels, scale data (e.g., z-score or min-max), and split into training/validation sets.
* Step 2. **Processing**: Use ABBA to symbolize time series, then tokenize with the LLM's tokenizer.
* Step 3. **Training**: Fine-tune the LLM on the tokenized dataset using LoRA and FSDP.
* Step 4. **Inference**: Generate predictions on new data, leveraging the fine-tuned model.

**Input Format**:
- **Training (``process`` method)**:
  - ``data``: Dictionary with:
    - ``'X_data'``: 2D NumPy array (n_samples × time_series_length, floats).
    - ``'Y_data'``: 1D NumPy array (n_samples, integers/strings for classification, floats for regression).
  - For forecasting, ``data`` is a 2D NumPy array of shape (time, features). It must contain at least twice the sum of ``seq_len_pre`` and ``seq_len_post`` so training and validation can use non-overlapping time ranges.
- **Inference**:
  - Classification and regression accept a 2D NumPy array (n_test_samples × time_series_length). Multivariate regression also supports batches of shape (samples, time, features).
  - Forecasting accepts one 2D history window of shape (time, features).

**Output**:
- **Classification/Regression**: A string with the generated prediction for a single sample, or a list of strings in input order for a batch.
- **Forecasting**: A 2D NumPy array (seq_len_post × n_features, floats) with the predicted time series continuation.

**Key Parameters**:
- **Initialization**:
  - ``abba_tol``, ``abba_alpha``, ``abba_init``, ``abba_scl``: ABBA parameters for symbolic approximation.
  - ``lora_r``, ``lora_alpha``, ``lora_dropout``: LoRA parameters for fine-tuning.
  - ``quant_process``: Enable 4-bit quantization (default: True).
- **Processing**:
  - ``project_name``: Identifier for saving models/scalers.
  - ``task``: 'classification', 'regression', or 'forecasting'.
  - ``prompt``: Task-specific instruction for the LLM.
  - ``scalar``: Scaling method ('z-score' or 'min-max').
  - ``alphabet_set``: Built-in alphabet selector, a list of whitespace-free custom symbols, or None to use symbols from the LLM tokenizer vocabulary.
  - ``seq_len_pre``/``seq_len_post``: Sequence lengths for forecasting input/output.
- **Training**:
  - ``num_epochs``: Number of training epochs.
  - ``output_dir``: Directory for saving checkpoints.
- **Inference**:
  - ``llm_max_new_tokens``: Maximum tokens for free-form regression and forecasting output. Classifiers trained by ``process()`` decode from their saved class vocabulary.
  - ``llm_temperature``: Controls randomness (0.0 for deterministic).
  - ``llm_repetition_penalty``: Penalizes repeated tokens.

**Example: ECG Classification with PTBDB**

This example uses `Mistral-7B-Instruct-v0.1 <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1>`_ to classify ECG signals. Classification uses the tokenizer's chat template when available, trains on the answer tokens, and decodes only the saved class labels. Run the example from the ``examples`` directory.

.. code-block:: python

   import pandas as pd
   import numpy as np
   from llmabba.llmabba import LLMABBA
   from sklearn.model_selection import train_test_split
   from transformers.trainer_utils import get_last_checkpoint

   project_name = "PTBDB"
   task_type = "classification"
   model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
   output_dir = '../save/PTBDB/checkpoints'
   prompt_input = (
       "Classify the ECG represented by the symbolic series. Use 0 for abnormal and 1 for normal. "
       "Reply with exactly one digit, 0 or 1."
   )

   # Load and process data
   abnormal_df = pd.read_csv('../test_data/ptbdb_abnormal.csv', header=None)
   normal_df = pd.read_csv('../test_data/ptbdb_normal.csv', header=None)

   Y_data = np.concatenate((np.zeros([abnormal_df.shape[0]], dtype=int), np.ones([normal_df.shape[0]], dtype=int)))
   # The last column of each CSV contains its label.
   X_data = pd.concat([abnormal_df, normal_df], ignore_index=True).iloc[:, :-1].to_numpy()

   train_data, test_data, train_target, test_target = train_test_split(
       X_data, Y_data, test_size=0.2, stratify=Y_data, random_state=42,
   )

   train_data, _, train_target, _ = train_test_split(
       train_data, train_target, train_size=500, stratify=train_target, random_state=42,
   )
   train_data_split = {'X_data': train_data, 'Y_data': train_target}

   # Train with LLM-ABBA
   LLMABBA_classification = LLMABBA()
   model_input, model_tokenizer = LLMABBA_classification.model(model_name=model_name, max_len=424)

   tokenized_train_dataset, tokenized_val_dataset = LLMABBA_classification.process(
       project_name=project_name,
       data=train_data_split,
       task=task_type,
       prompt=prompt_input,
       alphabet_set=-1,
       model_tokenizer=model_tokenizer,
       scalar="z-score"
   )

   LLMABBA_classification.train(
       model_input=model_input,
       num_epochs=1,
       output_dir=output_dir,
       train_dataset=tokenized_train_dataset,
       val_dataset=tokenized_val_dataset
   )

   # Inference
   test_data = np.expand_dims(test_data[1], axis=0)
   peft_model_input, model_tokenizer = LLMABBA_classification.model(
       peft_file=get_last_checkpoint(output_dir),
       model_name=model_name,
       max_len=424
   )

   out_text = LLMABBA_classification.inference(
       project_name=project_name,
       data=test_data,
       task=task_type,
       prompt=prompt_input,
       ft_model=peft_model_input,
       model_tokenizer=model_tokenizer,
   )

   print(out_text)

Saving and Loading Models
-------------------------

LLM-ABBA automatically saves the scaler and ABBA model during processing to ``../save/[project_name]/[task]_Scaler_save.pkl`` and ``../save/[project_name]/[task]_ABBA_save.pkl``. These are loaded during inference to ensure consistent scaling and symbolization.

**Example**:

.. code-block:: python

   from llmabba.llmabba import save_abba
   save_abba(LLMABBA_classification.xabba, "../save/PTBDB/classification_ABBA_save.pkl")

   # Loading
   from llmabba.llmabba import load_abba
   xabba = load_abba("../save/PTBDB/classification_ABBA_save.pkl")


Visualization
------------
Currently under development.

Common Questions
---------------
**1. Why are some LLM-generated tokens not recognized by ABBA?**

LLM hallucination may generate tokens outside ABBA’s alphabet. Solutions include:

- **Method A**: Skip or replace unrecognized tokens, using `repetition_penalty=1.3` and `temperature=0.0`.
- **Method B**: Use as many LLM tokens as possible (e.g., 32,000 for Mistral) to avoid this issue.
- **Method C**: Use ASCII symbols instead of LLM tokens, specifying them in the prompt (under testing).
- **Method D**: Fine-tune the LLM with `add_tokens` (not recommended).

**2. Is inhibition level necessary?**

Adjusting the inhibition level helps LLMs adapt to downstream tasks, especially with special symbols (Method C).

Contributing
------------
We welcome contributions in any form! Assistance with documentation is always welcome. To contribute, feel free to open an issue or please fork the project make your changes and submit a pull request. We will do our best to work through any issues and requests.

License
-------
Released under the MIT License.

Contact
-------
For questions, use GitHub issues or contact the maintainers or paper’s authors.

References
----------

Users of ``LLM-ABBA`` and ``ABBA`` as well as the software can cite:

.. bibliography:: bibtex
   
   @article{carson2024,
         author={Xinye Chen and Erin Carson and Cheng Kang},
         title={{LLM-ABBA}: Understanding time series via symbolic approximation}, 
         year={2026},
         journal = {Accepted in IEEE Transactions on Signal Processing},
         url={https://arxiv.org/abs/2411.18506}, 
   }
   
   @article{10.1145/3532622,
      author = {Chen, Xinye and G\"{u}ttel, Stefan},
      title = {An Efficient Aggregation Method for the Symbolic Representation of Temporal Data},
      year = {2023},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      volume = {17},
      number = {1},
      doi = {10.1145/3532622},
      journal = {ACM Trans. Knowl. Discov. Data},
      numpages = {22},
   }

   @article{Chen2024,
      author = {Chen, Xinye and Güttel, Stefan}, 
      title = {{fABBA}: A Python library for the fast symbolic approximation of time series}, 
      doi = {10.21105/joss.06294},
      year = {2024}, 
      publisher = {The Open Journal}, 
      volume = {9}, number = {95}, 
      pages = {6294}, 
      journal = {Journal of Open Source Software} 
   }




