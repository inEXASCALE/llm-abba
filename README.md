# LLM-ABBA

[![PyPI version][pypi-badge]][pypi] [![Documentation build][docs-badge]][docs] [![arXiv paper][paper-badge]][paper] [![Cython acceleration][cython-badge]][getting-started] [![MIT license][license-badge]][license]

LLM-ABBA is a Python library and research codebase for using large language models with numerical time series. It converts signals into sequences of symbols that a language model can process as text.

## What LLM-ABBA does

Adaptive Brownian bridge-based symbolic aggregation (ABBA) approximates a signal with variable-length linear segments. A fitted codebook groups segments with similar lengths and changes in value, then assigns each group a symbol. The resulting sequence can be used as model input or decoded into an approximation of the original signal.

- **Symbolic approximation and reconstruction** — Use `ABBA` to encode and decode time series on a CPU without downloading a language model.
- **Classification and regression** — Use `LLMABBA` to fine-tune a causal language model to predict class labels or numeric targets from symbolic inputs.
- **Forecasting** — Generate future symbols from a history window and decode them into a forecast in the original units.

## Who it is for

- **Time-series researchers** studying symbolic representations or comparing them across classification, regression, and forecasting tasks.
- **Machine learning engineers and data scientists** prototyping models for signals such as ECG recordings, sensor streams, or energy measurements.
- **Students and educators** exploring how piecewise linear approximation, symbolization, and language-model fine-tuning fit together.

## Installation

```bash
python -m pip install llmabba
```

See the [installation guide][getting-started] for source installation and GPU setup.

## Quick start

### Encode and reconstruct a time series

```python
from llmabba import ABBA

series = [[1.2, 1.4, 1.3, 1.8, 2.2, 2.4, 2.1]]
abba = ABBA(tol=0.1, alpha=0.1, verbose=0)
symbols = abba.encode(series)
reconstruction = abba.decode(symbols)
```

### Classify ECG signals with Mistral-7B

After [installing from source][getting-started], run the PTBDB demo from the repository root on a CUDA setup that supports bitsandbytes 4-bit quantization.

```bash
cd examples
WANDB_MODE=disabled python demo-ts-classification.py --model mistralai/Mistral-7B-Instruct-v0.1
```

## Documentation

- [Getting started][getting-started] covers installation, CUDA setup, and symbolic approximation.
- [PTBDB classification tutorial][classification] covers data splits, class balancing, and evaluation on the held-out test set.
- [Regression and forecasting][workflows] covers input shapes, generation, and checkpoint reuse.
- [API reference][api] covers the Python classes and methods.
- [References and citation][citation] provides publication details.

## Citation

If you use LLM-ABBA in research, cite the article in *IEEE Transactions on Signal Processing*. An [open-access preprint][paper] is available on arXiv.

```bibtex
@article{chen2026llmabba,
  author = {Chen, Xinye and Carson, Erin and Kang, Cheng},
  title = {{LLM-ABBA}: Understanding Time Series via Symbolic Approximation},
  journal = {IEEE Transactions on Signal Processing},
  year = {2026},
  volume = {74},
  pages = {1218--1232},
  doi = {10.1109/TSP.2026.3662011}
}
```

For the underlying aggregation method and its Python implementation, see [the fABBA method paper][fabba-method] and [the fABBA software paper][fabba-software].

<details>
<summary>BibTeX for the fABBA method and software</summary>

```bibtex
@article{10.1145/3532622,
  author = {Chen, Xinye and G\"{u}ttel, Stefan},
  title = {An Efficient Aggregation Method for the Symbolic Representation of Temporal Data},
  journal = {ACM Transactions on Knowledge Discovery from Data},
  year = {2023},
  volume = {17},
  number = {1},
  doi = {10.1145/3532622}
}

@article{Chen2024,
  author = {Chen, Xinye and G\"{u}ttel, Stefan},
  title = {{fABBA}: A Python library for the fast symbolic approximation of time series},
  journal = {Journal of Open Source Software},
  year = {2024},
  volume = {9},
  number = {95},
  pages = {6294},
  doi = {10.21105/joss.06294}
}
```

</details>

## Contributing and license

Report bugs or ask questions in [GitHub Issues][issues]. Documentation and code contributions are welcome. LLM-ABBA is released under the [MIT License][license].

[pypi-badge]: https://img.shields.io/pypi/v/llmabba?style=flat-square&logo=pypi&logoColor=white&label=PyPI&color=2563eb
[docs-badge]: https://img.shields.io/readthedocs/llm-abba/latest?style=flat-square&logo=readthedocs&logoColor=white&label=Docs
[paper-badge]: https://img.shields.io/badge/arXiv-2411.18506-b31b1b?style=flat-square&logo=arxiv&logoColor=white
[cython-badge]: https://img.shields.io/badge/Cython-accelerated-0f766e?style=flat-square
[license-badge]: https://img.shields.io/github/license/inEXASCALE/llm-abba?style=flat-square&color=64748b
[pypi]: https://pypi.org/project/llmabba/
[docs]: https://llm-abba.readthedocs.io/en/latest/
[paper]: https://arxiv.org/abs/2411.18506
[license]: https://github.com/inEXASCALE/llm-abba/blob/master/LICENSE
[getting-started]: https://llm-abba.readthedocs.io/en/latest/getting-started.html
[classification]: https://llm-abba.readthedocs.io/en/latest/classification.html
[workflows]: https://llm-abba.readthedocs.io/en/latest/workflows.html
[api]: https://llm-abba.readthedocs.io/en/latest/api.html
[citation]: https://llm-abba.readthedocs.io/en/latest/references.html
[fabba-method]: https://arxiv.org/abs/2201.05697
[fabba-software]: https://joss.theoj.org/papers/10.21105/joss.06294
[issues]: https://github.com/inEXASCALE/llm-abba/issues
