LLM-ABBA
========

|pip| |download status| |license| |download status2| 

.. |pip| image:: https://img.shields.io/pypi/v/llm-abba?color=red
   :target: https://github.com/inEXASCALE/llm-abba

.. |download status| image:: https://img.shields.io/pypi/v/llm-abba?color=red
   :target: /edit/master/README.rst

.. |license| image:: https://anaconda.org/conda-forge/classixclustering/badges/license.svg
   :target: https://github.com/inEXASCALE/llm-abba/blob/master/LICENSE

.. |download status2| image:: https://img.shields.io/pypi/dm/llm-abba.svg?label=PyPI%20downloads
   :target: https://pypi.org/project/llm-abba/

LLM-ABBA is an software framework designed for performing time series application using Large Language Models (LLMs) based on symbolic representation, as introduced in the paper:
`LLM-ABBA: Symbolic Time Series Approximation using Large Language Models <https://arxiv.org/abs/2411.18506>`_.

Time series analysis often involves identifying patterns, trends, and structures within sequences of data points. Traditional methods, such as discrete wavelet transforms or symbolic aggregate approximation (SAX), have demonstrated success in converting continuous time series into symbolic representations, facilitating better analysis and compression. However, these methods are often limited in their ability to capture complex and subtle patterns.

LLM-ABBA builds upon these techniques by incorporating the power of large language models, which have been shown to excel in pattern recognition and sequence prediction tasks. By applying LLMs to symbolic time series representation, LLM-ABBA is able to automatically discover rich, meaningful representations of time series data. This approach offers several advantages:

- **Higher accuracy and compression**: LLM-ABBA achieves better symbolic representations by leveraging LLMs' ability to understand and generate sequences, resulting in higher data compression and more accurate representation of underlying patterns.
- **Adaptability**: The use of LLMs enables the framework to adapt to various types of time series data, allowing for robust performance across different domains such as finance, healthcare, and environmental science.
- **Scalability**: LLM-ABBA is designed to efficiently handle large-scale time series datasets, making it suitable for both small and big data applications.
- **Automatic feature discovery**: By harnessing the power of LLMs, LLM-ABBA can discover novel features and patterns in time series data that traditional symbolic approaches might miss.

In summary, LLM-ABBA represents a significant advancement in symbolic time series analysis, combining the power of modern machine learning techniques with established methods to offer enhanced compression, pattern recognition, and interpretability.

Key Features
------------
- **Symbolic Time Series Approximation**: Converts time series data into symbolic representations.
- **LLM-Powered Encoding**: Utilizes LLMs to enhance compression and pattern discovery.
- **Efficient and Scalable**: Designed to work with large-scale time series datasets.
- **Flexible Integration**: Compatible with various machine learning and statistical analysis workflows.

Installation
------------
LLM-ABBA can be installed via pip:

.. code-block:: bash

    pip install llm-abba



Usage
-----
Here is a simple example of how to use LLM-ABBA:

.. code-block:: python

    from llm_abba import ABBA

    ts = [1.2, 1.4, 1.3, 1.8, 2.2, 2.4, 2.1]
    symbolic_representation = ABBA.encode(ts)

    print("Symbolic Representation:", symbolic_representation)

For more details, please refer to the documentation.

Contributing
------------
We welcome contributions! If you'd like to improve LLM-ABBA, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request.

License
-------
LLM-ABBA is released under the MIT License.

Contact
-------
For questions or feedback, please reach out via GitHub issues or contact the authors of the paper.



References
-----------
[1]Carson, E., Chen, X., and Kang, C., “LLM-ABBA: Understanding time series via symbolic approximation”, arXiv e-prints, arXiv:2411.18506, 2024. doi:10.48550/arXiv.2411.18506.
